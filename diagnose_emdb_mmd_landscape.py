"""Research-only sigma=1 MMD diagnosis for frozen EMD-9515 pair_017.

Reuse the benchmark objective, Autograd/Pymanopt derivatives and registration
adapter. Never fit ICP/CPD/Go-ICP, tune sigma, or write benchmark results.
"""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.optimizers import TrustRegions

import emdb_baseline_comparison as registration
import emdb_independent_resampling_comparison as benchmark
import generate_emdb_resampling_pairs as dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_9515_mmd_diagnosis_pair_017"
PAIR_ID = "pair_017"
STEPS = (1e-2, 1e-3, 1e-4, 1e-5)


def skew(u):
    x, y, z = np.asarray(u, dtype=np.float64)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def left_step(R, vector):
    """Exp(skew(vector)) @ R; vector components are radians."""
    return Rotation.from_rotvec(vector).as_matrix() @ R


def relative_log(start, end):
    return Rotation.from_matrix(end @ start.T).as_rotvec()


def tangent_basis(R_final, R_true):
    toward_truth = relative_log(R_final, R_true)
    if np.linalg.norm(toward_truth) < 1e-12:
        raise ValueError("Truth direction is undefined at coincident rotations.")
    u = toward_truth / np.linalg.norm(toward_truth)
    axis = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = axis - np.dot(axis, u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    np.testing.assert_allclose(np.stack((u, v, w)) @ np.stack((u, v, w)).T,
                               np.eye(3), atol=1e-14, rtol=0)
    return np.stack((u, v, w))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def gradient_norm(problem, R):
    return float(problem.manifold.norm(R, problem.riemannian_gradient(R)))


def gradient_checks(problem, objective, candidates, directions):
    """Use body-skew tangent representation and the actual Frobenius metric.

    dR/dh=skew(u)@R corresponds to body tangent R.T@skew(u)@R.
    A Euclidean unit rotation axis has metric tangent norm sqrt(2), not 1.
    No manually derived MMD gradient is substituted for Pymanopt's gradient.
    """
    rows = []
    for name, R in candidates.items():
        grad = problem.riemannian_gradient(R)
        for k, direction in enumerate(directions):
            body_tangent = R.T @ skew(direction) @ R
            implemented = float(problem.manifold.inner_product(R, grad, body_tangent))
            for h in STEPS:
                plus, minus = objective.evaluate(left_step(R, h*direction)), objective.evaluate(left_step(R, -h*direction))
                finite_difference = (plus-minus)/(2*h)
                discrepancy = abs(finite_difference-implemented)
                rows.append({"candidate": name, "direction": k, "axis_vector": json.dumps(direction.tolist()),
                             "h_radians": h, "metric_tangent_norm": float(problem.manifold.norm(R, body_tangent)),
                             "implemented_directional_derivative": implemented,
                             "finite_difference_derivative": finite_difference,
                             "absolute_discrepancy": discrepancy,
                             "relative_discrepancy": discrepancy/max(abs(implemented), abs(finite_difference), 1e-16)})
    # Practical check avoids treating relative error of a near-zero derivative
    # as a failure. Report every step including cancellation-prone small steps.
    practical = [r for r in rows if r["h_radians"] in (1e-3, 1e-4)]
    if any(r["absolute_discrepancy"] > 2e-10 + 1e-3*abs(r["implemented_directional_derivative"]) for r in practical):
        raise RuntimeError("Directional gradient check failed; no gradient implementation changed.")
    return rows


def refine(problem, objective, initial, truth, tolerance, config, label):
    """Same TrustRegions call as register_point_clouds; vary only final tolerance."""
    initial = initial.copy()
    initial_mmd = objective.evaluate(initial)
    initial_gradient = gradient_norm(problem, initial)
    optimizer = TrustRegions(max_iterations=config.max_iterations,
                             min_gradient_norm=tolerance, use_rand=False, verbosity=0)
    start = perf_counter()
    result = optimizer.run(problem, initial_point=initial)
    runtime = perf_counter()-start
    registration.assert_rotation(result.point)
    record = {"label": label, "start_role": "method initialization" if label.startswith("greedy") else "ORACLE/EXTERNAL diagnostic only",
              "min_gradient_norm": tolerance, "sigma": config.sigma,
              "initial_mmd": initial_mmd, "final_mmd": objective.evaluate(result.point),
              "mmd_decrease": initial_mmd-objective.evaluate(result.point),
              "initial_gradient_norm": initial_gradient, "final_gradient_norm": gradient_norm(problem, result.point),
              "iterations": result.iterations, "stopping_criterion": result.stopping_criterion,
              "displacement_degrees": registration.rotation_angle_error_degrees(initial, result.point),
              "angular_error_degrees": registration.rotation_angle_error_degrees(truth, result.point),
              "rotation_error": float(np.linalg.norm(truth-result.point)), "runtime_seconds": runtime,
              "R_initial": json.dumps(initial.tolist()), "R_final": json.dumps(result.point.tolist())}
    print(f"{label} tol={tolerance:.0e}: MMD={record['final_mmd']:.16e}, "
          f"angle={record['angular_error_degrees']:.9f}, grad={record['final_gradient_norm']:.4e}, "
          f"iterations={result.iterations}, runtime={runtime:.2f}s", flush=True)
    return record, result.point


def profile(objective, start, end, name):
    vector = relative_log(start, end)
    np.testing.assert_allclose(left_step(start, vector), end, atol=1e-12, rtol=0)
    base, endpoint = objective.evaluate(start), objective.evaluate(end)
    rows = [{"path": name, "t": float(t), "angle_from_start_degrees": float(np.degrees(abs(t)*np.linalg.norm(vector))),
             "mmd": objective.evaluate(left_step(start, t*vector))} for t in np.linspace(-.25, 1.25, 81)]
    minimum = min(rows, key=lambda r:r["mmd"])
    segment = [r for r in rows if 0 <= r["t"] <= 1]
    segment_minimum = min(segment, key=lambda r:r["mmd"])
    maximum = max([base, endpoint]+[r["mmd"] for r in segment])
    return rows, {"path": name, "start_mmd": base, "endpoint_mmd": endpoint,
                  "minimum_mmd": minimum["mmd"], "minimum_t": minimum["t"],
                  "segment_minimum_mmd": min(base, endpoint, segment_minimum["mmd"]),
                  "maximum_mmd_on_segment": maximum,
                  "maximum_mmd_sampled_full_range": max(r["mmd"] for r in rows),
                  "barrier_above_start_full_range": max(r["mmd"] for r in rows)-base,
                  "barrier_above_start_on_segment": maximum-base,
                  "barrier_above_higher_endpoint": maximum-max(base, endpoint),
                  "lower_region_away_from_start": minimum["mmd"] < base-1e-12,
                  "total_angle_degrees": float(np.degrees(np.linalg.norm(vector)))}


def local_landscape(objective, final, basis):
    values = np.radians(np.linspace(-20., 20., 21))
    rows = []
    for a in values:
        for b in values:
            R = left_step(final, a*basis[0]+b*basis[1])
            rows.append({"a_radians": float(a), "b_radians": float(b),
                         "a_degrees": float(np.degrees(a)), "b_degrees": float(np.degrees(b)),
                         "mmd": objective.evaluate(R)})
    curves = []
    base = objective.evaluate(final)
    for k, direction in enumerate(basis):
        for h in STEPS:
            plus, minus = objective.evaluate(left_step(final, h*direction)), objective.evaluate(left_step(final, -h*direction))
            curves.append({"direction": k, "axis_vector": json.dumps(direction.tolist()), "h_radians": h,
                           "curvature_per_radian_squared": (plus-2*base+minus)/h**2})
    # Full finite-difference Hessian adds evidence beyond a 2D slice or three
    # positive directional curvatures. It is still strictly local evidence.
    h = 1e-3
    H = np.empty((3, 3))
    for i in range(3):
        H[i, i] = next(r["curvature_per_radian_squared"] for r in curves if r["direction"]==i and r["h_radians"]==h)
        for j in range(i):
            a, b = h*basis[i], h*basis[j]
            H[i, j] = H[j, i] = (objective.evaluate(left_step(final, a+b))-objective.evaluate(left_step(final, a-b))
                                  -objective.evaluate(left_step(final, -a+b))+objective.evaluate(left_step(final, -a-b)))/(4*h*h)
    return rows, curves, {"h_radians": h, "hessian_rotation_vector_coordinates": H.tolist(),
                          "hessian_eigenvalues": np.linalg.eigvalsh(H).tolist(),
                          "metric_note": "Unit Euclidean rotation axes have Frobenius metric norm sqrt(2); metric-unit Hessian eigenvalues are half these."}


def plots(output, profiles, landscape, basis, final, truth, goicp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = next(r["mmd"] for r in landscape if abs(r["a_radians"])+abs(r["b_radians"])<1e-12)
    for name, rows in profiles.items():
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot([r["t"] for r in rows], [1e6*(r["mmd"]-base) for r in rows])
        ax.axvline(0, color="gray", ls="--", label="Ours final")
        ax.axvline(1, color="#db7621", ls="--", label="R_true" if name=="truth" else "Go-ICP SO(3) rotation")
        ax.axhline(0, color="gray", lw=.7)
        ax.set(xlabel="Geodesic parameter t", ylabel="MMD minus Ours final (x 1e6)",
               title=f"pair_017, sigma=1: Ours final to {name}")
        ax.grid(alpha=.25); ax.legend()
        fig.savefig(output/f"mmd_geodesic_to_{name}.png", dpi=180); plt.close(fig)
    axis = sorted({r["a_degrees"] for r in landscape})
    grid = np.array([r["mmd"] for r in landscape]).reshape(21, 21)
    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    contours = ax.contourf(axis, axis, 1e6*(grid.T-base), levels=24, cmap="viridis")
    fig.colorbar(contours, ax=ax, label="MMD minus Ours final (x 1e6)")
    ax.scatter([0], [0], marker="x", s=85, color="white", label="Ours final")
    positions = {}
    for name, endpoint in (("R_true", truth), ("R_goicp", goicp)):
        vector = relative_log(final, endpoint)
        coordinates = basis @ vector
        residual = float(abs(coordinates[2]))
        in_slice = residual < 1e-10 and np.max(np.abs(coordinates[:2])) <= np.radians(20.)
        positions[name] = {"coordinates_radians": coordinates.tolist(), "out_of_plane_residual_radians": residual,
                           "marked_on_contour": bool(in_slice)}
        if in_slice:
            a, b = np.degrees(coordinates[:2])
            ax.scatter([a], [b], marker="*", s=130, label=name, edgecolors="black")
    ax.set(xlabel="a: toward truth (degrees)", ylabel="b: perpendicular direction (degrees)",
           title="pair_017: local exponential-coordinate slice, sigma=1", aspect="equal")
    ax.legend(); fig.savefig(output/"mmd_local_contour.png", dpi=180); plt.close(fig)
    return positions


def table(rows, fields):
    def fmt(value):
        return f"{value:.12e}" if isinstance(value, float) else str(value).replace("|", "/")
    return "\n".join(["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields)+" |"]+
                     ["| " + " | ".join(fmt(row[field]) for field in fields) + " |" for row in rows])


def report(output, reproduction, candidates, checks, paths, curves, H, thresholds, alternatives, positions):
    baseline = next(r for r in candidates if r["candidate"]=="R_final")
    truth = next(r for r in candidates if r["candidate"]=="R_true")
    goicp = next(r for r in candidates if r["candidate"]=="R_goicp")
    strict = thresholds[-1]
    best_alt = min(alternatives, key=lambda r:r["final_mmd"])
    strict_R = np.array(json.loads(strict["R_final"]))
    max_alt_distance = max(float(np.linalg.norm(np.array(json.loads(r["R_final"]))-strict_R)) for r in alternatives)
    final_R = np.array(json.loads(baseline["R"]))
    strict_displacement = float(np.degrees(np.linalg.norm(relative_log(final_R,strict_R))))
    loss_gain = baseline["mmd"]-strict["final_mmd"]
    truth_gap = truth["mmd"]-baseline["mmd"]
    same_solution = max_alt_distance < 1e-7
    if same_solution and truth_gap > 1e-10 and goicp["mmd"] > baseline["mmd"] and loss_gain < truth_gap*1e-3:
        conclusion = f"""The dominant observed limitation is empirical-objective mismatch (C).
The stricter solve, truth start and external Go-ICP start converge to essentially
the same rotation (largest external-vs-strict Frobenius difference
{max_alt_distance:.16e}) at about {strict['angular_error_degrees']:.9f} degrees
from the imposed truth. Starting at truth actually moves away from it while
lowering MMD. The truth/Go-ICP orientations have higher MMD by approximately
2.1e-5, so these checks provide no evidence of a better-MMD basin near truth (B).

Early termination (A) has only a minor effect here: the stricter solution moves
{strict_displacement:.12f} degrees from the saved final rotation and lowers MMD
by {loss_gain:.16e}, only {loss_gain/truth_gap:.8e} of the truth-vs-final loss
gap. Its angular error slightly increases. A stricter stopping threshold therefore
does not explain or fix the approximately 13.4-degree recovery error.

The landscape is anisotropic, with a weaker direction toward truth, but the
positive local Hessian eigenvalues and successful derivative checks do not
support a purely floating-point-flat explanation. This is local evidence, not
a global-optimum proof or a statistical identifiability test. Two independent
finite clouds need not have their empirical MMD minimum exactly at the imposed
population rotation. The underlying cause could involve sampling variability
and bandwidth-dependent sensitivity; this single-pair experiment cannot
separate those effects or establish consistency/inconsistency in the population.
"""
    else:
        conclusion = """The diagnostics do not support a single automatic classification.
Use the candidate loss gaps, strict-start improvement, local Hessian and oracle
solutions together. A global optimum and statistical cause remain unestablished.
"""
    practical = [r for r in checks if r["h_radians"] in (1e-3, 1e-4)]
    text = f"""# MMD landscape diagnosis: EMD-9515 independent-resampling pair_017

This is a single-pair research diagnostic, not a benchmark or algorithm change.
All fitting here uses sigma=1. Go-ICP/ICP/CPD were NOT rerun. Saved Go-ICP
translation is excluded from every rotation-only MMD comparison.

## Exact reproduction

The original `emdb_baseline_comparison.run_ours` adapter was called with frozen
X/Z, stored R_true and `default_rng(ours_fit_seed)` after the unchanged
`emdb_independent_resampling_comparison.prepare_pair` preprocessing. Reproduction
passed before any diagnostic sweeps: rotation Frobenius difference
{reproduction['R_final_fro_difference']:.16e}, angular-error difference
{reproduction['angular_error_difference_degrees']:.16e}, final MMD difference
{reproduction['final_mmd_difference']:.16e}. Angle is
{reproduction['angular_error_degrees']:.12f} degrees; iterations
{reproduction['iterations']}; gradient {reproduction['gradient_norm']:.16e}.
Both geometric frames and the greedy rotation are returned by the unchanged
registration implementation, so the intermediate initializations are recovered
unambiguously. `R_pca_initial=U_Y@U_X.T`; greedy signs
{reproduction['greedy_signs']}; `R_greedy_initial=(U_Y*signs)@U_X.T`.
Matrices and reproduction details are in reproduction_check.json/rotations.json.

The saved Go-ICP matrix has orthogonality residual
{reproduction['goicp_raw_orthogonality_error_fro']:.16e}; determinant
{reproduction['goicp_raw_determinant']:.16e}. For exact SO(3) geodesics/tangent
analysis and external-start refinement only, an explicit nearest-proper-SO(3)
SVD projection changes it by {reproduction['goicp_geometry_projection_distance']:.16e}
in Frobenius norm. The saved raw matrix is preserved in rotations.json and its
rotation-only MMD is {reproduction['goicp_raw_rotation_only_mmd']:.16e}.
The primary candidate table uses the proper rotation; no benchmark CSV or
Go-ICP cleanup policy was changed. No transpose/inverse was applied.
The additional R_goicp_saved_raw row also reports the exact stored matrix's
MMD, benchmark angle and implemented Pymanopt projected-gradient norm. Because
that raw matrix is slightly outside exact SO(3), its gradient norm is an
approximate numerical diagnostic, not an exact intrinsic gradient at an SO(3)
point. The saved raw benchmark angle is
{reproduction['goicp_raw_benchmark_angle_degrees']:.12f} degrees, versus
{goicp['angular_error_degrees']:.12f} degrees after geometry-only cleanup.
This trace/arccos metric is sensitive to tiny orthogonality drift near zero;
an exact truth-vs-truth call can likewise return a microdegree rounding residual.

## Exact objective and metric

Reuse `emdb_baseline_comparison.RotationMMD.evaluate/.problem`: the original
diagonal-excluding within-cloud MMD terms and cross-kernel mean, with
exp(-squared_distance/(2*sigma^2)). The Autograd cost equals the evaluate
implementation on SO(3); candidate consistency checks and full mmd_squared
checks passed. Negative MMD values are allowed by this estimator.
The same implementation in numerical_experiments.py was inspected; no alternate
estimator, bandwidth, objective scaling or manually derived gradient was used.

Pymanopt 2.2.0 represents tangent vectors as body-skew matrices; its metric is
the Frobenius inner product and its exponential is R@Exp(body_tangent).
For the requested left curve Exp(h*skew(u))@R, the comparison tangent is
R.T@skew(u)@R. Unit Euclidean axis vectors have metric norm sqrt(2).

## Candidate comparison

{table(candidates, ['candidate','mmd','mmd_minus_final','gradient_norm','angular_error_degrees','distance_from_final_degrees'])}

## Gradient finite-difference validation

Central differences in double precision used three orthogonal axes (first toward
truth), at h=0.01, 0.001, 0.0001 and 0.00001 radians for greedy, final and truth.
All 36 absolute/relative discrepancies are in gradient_checks.csv. For the
requested practical h=1e-3/1e-4, maximum absolute discrepancy is
{max(r['absolute_discrepancy'] for r in practical):.16e}; maximum relative
discrepancy is {max(r['relative_discrepancy'] for r in practical):.16e}.
The derivative checks passed; no gradient formula was changed. Relative errors
near zero derivatives can be sensitive to cancellation; raw derivatives and
absolute discrepancies remain available.

## Geodesics and local landscape

Paths use Exp(t*Log(R_endpoint@R_final.T))@R_final at 81 points on
[-0.25,1.25]. This sampling does not hit t=0 or t=1 exactly; endpoint values
were evaluated separately. Barriers below are computed on [0,1], with exact
endpoints included. Values are sampled evidence, not certified extrema.

{table(paths, ['path','start_mmd','endpoint_mmd','minimum_mmd','minimum_t','barrier_above_start_on_segment','barrier_above_higher_endpoint','lower_region_away_from_start'])}

Extended [-0.25,1.25] sampled maxima/barriers are also recorded:

{table(paths, ['path','maximum_mmd_sampled_full_range','barrier_above_start_full_range'])}

The 21x21 local grid uses Exp(skew(a*u+b*v))@R_final with a,b in +/-20
degrees. Its coordinates are recorded in local_landscape.csv; truth is marked
at its actual in-plane coordinates. Go-ICP out-of-plane residual is
{positions['R_goicp']['out_of_plane_residual_radians']:.16e} radians and marked
on the contour: {positions['R_goicp']['marked_on_contour']}. It is never projected
onto the displayed tangent plane merely for plotting. Plot y-values/color are
(L-L(R_final))*1e6 for readability only; solver/objective are not rescaled.

Directional curvatures (per radian squared) for three Euclidean unit axes:

{table([r for r in curves if r['h_radians'] in (1e-3,1e-4)], ['direction','h_radians','curvature_per_radian_squared'])}

The finite-difference full 3x3 local Hessian at h=1e-3 has eigenvalues
{H['hessian_eigenvalues']} in rotation-vector coordinates. For metric-unit
axes these eigenvalues divide by 2. Curvature checks across step sizes are in
directional_curvatures.csv. A slice, positive directional curvatures or a local
positive-definite Hessian cannot establish a global optimum.

## Final-gradient threshold experiment

PCA and greedy are frozen. All three final TrustRegions runs start at exactly
the recovered greedy matrix; only min_gradient_norm varies. max_iterations=100,
use_rand=False, and every other default remain unchanged. Complete stopping
messages/runtime and matrices are in optimizer_threshold_comparison.csv.

{table(thresholds, ['min_gradient_norm','initial_mmd','final_mmd','initial_gradient_norm','final_gradient_norm','iterations','displacement_degrees','angular_error_degrees','runtime_seconds'])}

## ORACLE/EXTERNAL diagnostic starts

These runs are not legitimate initializations for the proposed method and
never enter benchmark results. Each uses the same final TrustRegions settings
with min_gradient_norm=1e-10; no oracle PCA or sign search is performed.

{table(alternatives, ['label','min_gradient_norm','initial_mmd','final_mmd','final_gradient_norm','iterations','displacement_degrees','angular_error_degrees','stopping_criterion'])}

## Evidence and interpretation

- Early stopping effect: the strict greedy-start run decreases the saved final
  MMD by {baseline['mmd']-strict['final_mmd']:.16e}, moves
  {strict['displacement_degrees']:.12f} degrees from initialization, and has
  {strict['angular_error_degrees']:.12f} degrees error. Check its final gradient
  and stopping criterion, rather than assuming its iteration count means success.
- Local-basin evidence: best external/oracle-start final MMD minus strict
  greedy-start MMD is {best_alt['final_mmd']-strict['final_mmd']:.16e}; compare
  final rotations and error, not native Go-ICP SSE. No global optimality claim
  follows from these two external starts.
- Empirical-objective preference: truth minus saved Ours MMD is
  {truth['mmd']-baseline['mmd']:.16e}; proper Go-ICP minus saved Ours MMD is
  {goicp['mmd']-baseline['mmd']:.16e}. A lower MMD need not recover the imposed
  population rotation better on two independently resampled finite clouds.
- Flatness/identifiability: inspect the small directional curvatures, Hessian
  spectrum and plotted loss changes against double-precision derivative checks.
  Numerical flatness is different from proven statistical uncertainty; this one
  pair provides no sampling confidence intervals.

## Dominant limitation and next scientific experiment

{conclusion}

First investigate sampling stability at the fixed original sigma=1: generate
independent evaluation clouds with predeclared seeds from the same thresholded
population, compare rotation-only held-out MMD at the already fixed rotations,
and repeat with larger point counts. Separate training and evaluation draws;
allow population overlap under the unchanged sampling protocol. Report
uncertainty across resamples rather than interpreting a single loss difference
as a population claim. This recommendation is not executed in this task.

After that, a predeclared bandwidth/sensitivity study could test whether sigma=1
emphasizes weak shape directions after joint normalization; select settings
through held-out criteria, not lowest error against R_true. These observations
give little reason to prioritize better initialization for this pair, since
external/oracle starts reached the same solution. Objective rescaling or tighter
tolerances may improve stopping robustness on other pairs, but they do not fix
this pair's observed objective preference. Do not modify the benchmark using
these oracle results.

## Preservation and limitations

input_metadata.json records source/manifest/result hashes, library versions,
thread environment, settings and basis. protected_file_hashes.json and
preservation_check.json verify old scripts, frozen dataset and saved benchmark
files were not edited. No full EMDB benchmark was launched. The original PCA
frame error concerns equivalent axes; registration error concerns the actual
imposed rotation. The finite sample objective is not a guarantee of population
rotation recovery. The diagnosis covers one pair, sigma=1, a finite slice and
these starts; no claims about all maps, bandwidths or global convergence.
"""
    (output/"REPORT.md").write_text(text, encoding="utf-8")


def run(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be new/empty; existing results will not be overwritten.")
    started = perf_counter()
    manifest, trial_csv = benchmark.DEFAULT_MANIFEST, benchmark.DEFAULT_OUTPUT/"baseline_trial_results.csv"
    saved_metadata = json.loads((trial_csv.parent/"baseline_run_metadata.json").read_text())
    signature = saved_metadata["reproducibility_signature"]
    if signature["registration_source_sha256"] != dataset.sha256(ROOT/"emdb_baseline_comparison.py"):
        raise ValueError("Registration code differs from completed benchmark; stop.")
    if signature["manifest_sha256"] != dataset.sha256(manifest) or signature["method_settings"] != benchmark.method_settings():
        raise ValueError("Manifest or settings differ from completed benchmark; stop.")
    versions = {name: metadata.version(name) for name in signature["package_versions"]}
    threads = {name: os.environ.get(name) for name in signature["thread_environment"]}
    if versions != signature["package_versions"] or threads != signature["thread_environment"]:
        raise ValueError("Package versions/thread environment differ from completed benchmark; stop.")
    pair = next(r for r in dataset.read_manifest(manifest) if r["pair_id"]==PAIR_ID)
    saved = {r["method"]:r for r in dataset.read_manifest(trial_csv) if r["pair_id"]==PAIR_ID}
    if set(saved) != set(registration.METHODS):
        raise ValueError("Expected all four saved method rows for pair_017.")
    for field, hash_field in (("source_file","source_sha256"),("target_unrotated_file","target_sha256")):
        if dataset.sha256(dataset.resolve_cloud(manifest,pair[field])) != pair[hash_field]:
            raise ValueError("Frozen pair file hash mismatch.")
    Xc, Yc, truth, scale = benchmark.prepare_pair(pair, manifest)
    for row in saved.values():
        if (row["Xc_sha256"] != benchmark.array_hash(Xc) or row["Yc_sha256"] != benchmark.array_hash(Yc)
                or row["R_true"] != pair["R_true"] or int(row["ours_fit_seed"]) != int(pair["ours_fit_seed"])
                or float(row["normalization_scale"]) != scale):
            raise ValueError("Saved input hashes, rotation, seed or preprocessing mismatch.")
    protected = [p for p in ROOT.glob("*.py") if p.name != Path(__file__).name]+list(manifest.parent.iterdir())+list(trial_csv.parent.rglob("*"))
    hashes = {str(p.resolve()):dataset.sha256(p) for p in protected if p.is_file()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"protected_file_hashes.json", hashes)
    print("Reproducing original Ours adapter and checking saved pair_017...", flush=True)
    result = registration.run_ours(Xc,Yc,np.random.default_rng(int(pair["ours_fit_seed"])),benchmark.CONFIG).registration
    final, greedy, pca = result.R_est, result.R_init, result.U_Y@result.U_X.T
    raw_goicp = np.array(json.loads(saved["goicp"]["R_est"]), dtype=np.float64)
    registration.assert_rotation(raw_goicp)
    # Explicit geometry-only numerical correction; never modifies a saved row.
    U, s, Vt = np.linalg.svd(raw_goicp)
    correction = np.eye(3); correction[-1,-1] = np.sign(np.linalg.det(U@Vt))
    goicp = U@correction@Vt
    registration.assert_rotation(goicp, atol=1e-12)
    manifold = SpecialOrthogonalGroup(3)
    objective = registration.RotationMMD(Xc,Yc,benchmark.CONFIG.sigma)
    problem = objective.problem(manifold)
    angle = registration.rotation_angle_error_degrees(truth, final)
    frame_error, _ = registration.signed_permutation_frame_error(truth@result.U_X,result.U_Y)
    reproduction = {"R_final_fro_difference":float(np.linalg.norm(final-np.array(json.loads(saved['ours']['R_est'])))),
                    "angular_error_difference_degrees":angle-float(saved['ours']['rotation_angle_error_degrees']),
                    "final_mmd_difference":result.final_mmd-float(saved['ours']['final_mmd']),
                    "angular_error_degrees":angle,"frame_error":frame_error,
                    "iterations":result.optimization.iterations,"gradient_norm":result.optimization.gradient_norm,
                    "greedy_signs":result.greedy.signs.tolist(),"greedy_accepted_pairs":result.greedy.accepted_pairs,
                    "greedy_history":result.greedy.history,"normalization_scale":scale,
                    "goicp_raw_orthogonality_error_fro":float(np.linalg.norm(raw_goicp.T@raw_goicp-np.eye(3))),
                    "goicp_raw_determinant":float(np.linalg.det(raw_goicp)),"goicp_raw_singular_values":s.tolist(),
                    "goicp_geometry_projection_distance":float(np.linalg.norm(goicp-raw_goicp)),
                    "goicp_raw_rotation_only_mmd":objective.evaluate(raw_goicp),
                    "goicp_raw_benchmark_angle_degrees":registration.rotation_angle_error_degrees(truth,raw_goicp)}
    passed = (reproduction["R_final_fro_difference"]<1e-9 and abs(reproduction["angular_error_difference_degrees"])<1e-7
              and abs(reproduction["final_mmd_difference"])<1e-12 and result.optimization.iterations==int(saved['ours']['winner_iterations'])
              and abs(frame_error-float(saved['ours']['frame_error']))<1e-9
              and abs(result.optimization.gradient_norm-float(saved['ours']['winner_gradient_norm']))<1e-12)
    reproduction["passed"] = bool(passed)
    write_json(output/"reproduction_check.json",reproduction)
    if not passed:
        raise RuntimeError("Original benchmark reproduction FAILED. STOP; no landscape/refinement tests run.")
    print(f"Exact reproduction passed: {angle:.12f} degrees; matrix difference {reproduction['R_final_fro_difference']:.3e}", flush=True)
    rotations = {"R_pca_initial":pca,"R_greedy_initial":greedy,"R_final":final,"R_true":truth,"R_goicp":goicp}
    write_json(output/"rotations.json",{**{k:v.tolist() for k,v in rotations.items()},"R_goicp_saved_raw":raw_goicp.tolist(),
                                       "U_X":result.U_X.tolist(),"U_Y":result.U_Y.tolist()})
    candidates=[]
    for name,R in rotations.items():
        registration.assert_rotation(R)
        value=objective.evaluate(R)
        np.testing.assert_allclose(problem.cost(R),value,atol=2e-15,rtol=0)
        np.testing.assert_allclose(registration.mmd_squared(Xc@R.T,Yc,1.),value,atol=2e-15,rtol=0)
        candidates.append({"candidate":name,"mmd":value,"mmd_minus_final":value-objective.evaluate(final),
                           "gradient_norm":gradient_norm(problem,R),"angular_error_degrees":registration.rotation_angle_error_degrees(truth,R),
                           "distance_from_final_degrees":registration.rotation_angle_error_degrees(final,R),"R":json.dumps(R.tolist()),
                           "rotation_note":"proper SO(3); Go-ICP explicitly cleaned for geometry" if name=="R_goicp" else "reproduced proper SO(3)"})
    candidates.append({"candidate":"R_goicp_saved_raw","mmd":objective.evaluate(raw_goicp),
                       "mmd_minus_final":objective.evaluate(raw_goicp)-objective.evaluate(final),
                       "gradient_norm":gradient_norm(problem,raw_goicp),
                       "angular_error_degrees":registration.rotation_angle_error_degrees(truth,raw_goicp),
                       "distance_from_final_degrees":registration.rotation_angle_error_degrees(final,raw_goicp),
                       "R":json.dumps(raw_goicp.tolist()),"rotation_note":"exact saved near-SO(3) matrix; projected gradient norm is approximate"})
    write_csv(output/"candidate_rotations.csv",candidates)
    basis=tangent_basis(final,truth)
    write_json(output/"input_metadata.json",{"pair_id":PAIR_ID,"manifest":str(manifest),"manifest_sha256":dataset.sha256(manifest),
               "trial_csv":str(trial_csv),"trial_csv_sha256":dataset.sha256(trial_csv),"pair":pair,"package_versions":versions,
               "diagnostic_script_sha256":dataset.sha256(Path(__file__)),
               "thread_environment":threads,"method_settings":benchmark.method_settings(),"dtype":str(Xc.dtype),
               "left_rotation_axis_basis_uvw":basis.tolist(),"goicp_translation_excluded":True,
               "goicp_projection_scope":"explicit geometry-only correction; preserve saved raw rotation and benchmark policy",
               "trustregions_default_settings":{"miniter":3,"kappa":.1,"theta":1.,"rho_prime":.1,"use_rand":False,
                  "rho_regularization":1000.,"max_time":1000,"min_step_size":1e-10,"max_cost_evaluations":5000}})
    checks=gradient_checks(problem,objective,{name:rotations[name] for name in ("R_greedy_initial","R_final","R_true")},basis)
    write_csv(output/"gradient_checks.csv",checks)
    print("Directional gradient checks passed. Evaluating 81-point geodesics and 21x21 local grid...",flush=True)
    profiles,paths={},[]
    for name,endpoint in (("truth",truth),("goicp",goicp)):
        profiles[name],summary=profile(objective,final,endpoint,name);paths.append(summary)
    write_csv(output/"geodesic_profiles.csv",profiles["truth"]+profiles["goicp"])
    write_json(output/"geodesic_summary.json",paths)
    landscape,curves,H=local_landscape(objective,final,basis)
    write_csv(output/"local_landscape.csv",landscape)
    write_csv(output/"directional_curvatures.csv",curves)
    write_json(output/"local_hessian.json",H)
    positions=plots(output,profiles,landscape,basis,final,truth,goicp)
    write_json(output/"tangent_slice_positions.json",positions)
    print("Landscape plots saved. Comparing final TrustRegions thresholds from the same frozen greedy start...",flush=True)
    thresholds=[]
    for tol in (1e-6,1e-8,1e-10):
        row,point=refine(problem,objective,greedy,truth,tol,benchmark.CONFIG,"greedy")
        thresholds.append(row)
        if tol==1e-6:
            np.testing.assert_allclose(point,final,atol=1e-12,rtol=0)
        write_csv(output/"optimizer_threshold_comparison.csv",thresholds)
    alternatives=[]
    for name,start in (("R_goicp_EXTERNAL",goicp),("R_true_ORACLE",truth)):
        row,_=refine(problem,objective,start,truth,1e-10,benchmark.CONFIG,name)
        alternatives.append(row);write_csv(output/"alternative_initializations.csv",alternatives)
    report(output,reproduction,candidates,checks,paths,curves,H,thresholds,alternatives,positions)
    changed=[p for p,digest in hashes.items() if dataset.sha256(Path(p))!=digest]
    write_json(output/"preservation_check.json",{"passed":not changed,"protected_files":len(hashes),"changed_files":changed,
               "diagnosis_elapsed_seconds":perf_counter()-started,"real_baselines_rerun":False,"bandwidth_tuned":False})
    if changed:
        raise RuntimeError("Protected file hash changed.")
    print(f"Finished diagnosis in {perf_counter()-started:.2f}s; outputs: {output}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    run(args.output_dir.resolve())


if __name__=="__main__":
    main()
