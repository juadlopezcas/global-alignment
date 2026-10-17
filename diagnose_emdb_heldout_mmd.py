"""Held-out sigma=1 MMD differences for FIXED EMD-9515 pair_017 rotations.

No registration/optimization, bandwidth choice or fitted-rotation scaling.
Exact float64 cross-kernel sums in two-dimensional tiles; the original
diagonal-excluding within terms cancel when taking rotation loss differences.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

import emdb_baseline_comparison as registration
import emdb_independent_resampling_comparison as benchmark
import generate_emdb_resampling_pairs as sampling

ROOT = Path(__file__).resolve().parent
DIAGNOSIS = ROOT / "numerical_results/emdb_9515_mmd_diagnosis_pair_017"
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_9515_heldout_mmd_pair_017"
DESIGN = ((1000, 20), (5000, 10), (10000, 5))
SIGMA = 1.0
DEFAULT_SEED = 20261007
SEED_NAMESPACE = 0x48454C44  # 'HELD', distinct from [42, pair_index] training.
CONTRASTS = {"Delta_ours_true": ("ours", "true"),
             "Delta_goicp_true": ("goicp", "true"),
             "Delta_ours_goicp": ("ours", "goicp")}
LABELS = {"Delta_ours_true": "Ours vs True", "Delta_goicp_true": "Go-ICP vs True",
          "Delta_ours_goicp": "Ours vs Go-ICP"}
NUMERICAL_TOLERANCE = 3e-14


def load_rotations():
    """Load the exact saved diagnosis candidates; never regenerate/refine them."""
    saved = json.loads((DIAGNOSIS / "rotations.json").read_text())
    check = json.loads((DIAGNOSIS / "reproduction_check.json").read_text())
    if not check["passed"]:
        raise ValueError("Previous pair_017 reproduction did not pass.")
    rotations = {"ours": np.array(saved["R_final"], dtype=np.float64),
                 "true": np.array(saved["R_true"], dtype=np.float64),
                 "goicp": np.array(saved["R_goicp"], dtype=np.float64)}
    raw = np.array(saved["R_goicp_saved_raw"], dtype=np.float64)
    U, _, Vt = np.linalg.svd(raw)
    correction = np.eye(3); correction[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    np.testing.assert_allclose(rotations["goicp"], U @ correction @ Vt, atol=1e-15, rtol=0)
    for R in rotations.values():
        registration.assert_rotation(R, atol=1e-12)
        R.setflags(write=False)
    return rotations, {"source": str(DIAGNOSIS / "rotations.json"),
                       "source_sha256": sampling.sha256(DIAGNOSIS / "rotations.json"),
                       "R_goicp_saved_raw": raw.tolist(),
                       "goicp_projection_reused": True,
                       "goicp_projection_needed_for_exact_SO3": not np.allclose(raw.T @ raw, np.eye(3), atol=1e-12, rtol=0),
                       "goicp_projection_distance": float(np.linalg.norm(rotations["goicp"] - raw)),
                       "policy": "reuse exact previously saved proper matrix, not refit or reproject the candidate",
                       "rotations": {name: R.tolist() for name, R in rotations.items()}}


def sampling_plan(master_seed=DEFAULT_SEED):
    """Predeclare all independent integer streams before any objective evaluation."""
    if master_seed < 0:
        raise ValueError("Seed must be nonnegative.")
    forbidden = {int(row[name]) for row in sampling.read_manifest(benchmark.DEFAULT_MANIFEST)
                 for name in sampling.STREAM_NAMES}
    seen, rows = set(), []
    for n, repetitions in DESIGN:
        for repetition in range(repetitions):
            entropy = [master_seed, SEED_NAMESPACE, n, repetition]
            children = np.random.SeedSequence(entropy).spawn(2)
            seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
            if any(seed in forbidden or seed in seen for seed in seeds) or seeds[0] == seeds[1]:
                raise ValueError("Seed collision with training/held-out stream; stop without adaptive reseeding.")
            seen.update(seeds)
            rows.append({"evaluation_id": f"n{n}_rep{repetition:03d}", "num_points": n,
                         "repetition": repetition, "planned_repetitions": repetitions,
                         "master_seed": master_seed, "seed_entropy": json.dumps(entropy),
                         "source_sampling_seed": seeds[0], "target_sampling_seed": seeds[1]})
    return rows


def sample_pair(population, plan, R_true):
    """Same RNG choice and exact %.4f ASCII round-trip as frozen benchmark XYZ."""
    clouds, hashes = [], []
    n = int(plan["num_points"])
    for key in ("source_sampling_seed", "target_sampling_seed"):
        points = sampling.sample_cloud(population, np.random.default_rng(int(plan[key])), n)
        serialized = sampling.xyz_bytes(points)
        cloud = np.loadtxt(io.BytesIO(serialized), dtype=np.float64)
        if cloud.shape != (n, 3) or not np.isfinite(cloud).all() or len(sampling.point_set(cloud)) != n:
            raise ValueError("Invalid held-out cloud shape, coordinates or duplicates.")
        clouds.append(cloud)
        hashes.append(hashlib.sha256(serialized).hexdigest())
    X, Z = clouds
    if hashes[0] == hashes[1]:
        raise ValueError("Accidentally identical held-out X/Z; no redraw.")
    Y = Z @ R_true.T
    Xc, Yc, scale = registration.center_and_normalize(X, Y)  # ONCE per pair.
    Xc.setflags(write=False); Yc.setflags(write=False)
    return Xc, Yc, {"X_xyz_sha256": hashes[0], "Z_xyz_sha256": hashes[1],
                    "Xc_sha256": benchmark.array_hash(Xc), "Yc_sha256": benchmark.array_hash(Yc),
                    "normalization_scale": scale,
                    "source_target_overlap_count": len(sampling.point_set(X) & sampling.point_set(Z))}


def blocked_cross_mean(X_rotated, Y, chunk_size=512):
    """Exact all n*m RBF pairs; only chunk_size x chunk_size matrices exist.

    Same distance algebra as RotationMMD.problem, no clipping/approximation.
    Tile sums and math.fsum accumulate in double precision. Within terms are
    not computed here because they cancel exactly in the requested contrasts.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    X_rotated, Y = np.asarray(X_rotated, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    x_sq = np.sum(X_rotated**2, axis=1)
    y_sq = np.sum(Y**2, axis=1)
    tile_sums = []
    for i in range(0, len(X_rotated), chunk_size):
        X_block = X_rotated[i:i+chunk_size]
        for j in range(0, len(Y), chunk_size):
            distances = (x_sq[i:i+chunk_size, None] + y_sq[None, j:j+chunk_size]
                         - 2 * X_block @ Y[j:j+chunk_size].T)
            distances *= -1 / (2 * SIGMA**2)
            np.exp(distances, out=distances)
            tile_sums.append(float(distances.sum(dtype=np.float64)))
    return math.fsum(tile_sums) / (len(X_rotated) * len(Y))


def evaluate_pair(Xc, Yc, rotations, chunk_size=512):
    """All candidates receive identical Xc/Yc; no candidate-specific normalization."""
    before = (benchmark.array_hash(Xc), benchmark.array_hash(Yc))
    means = {name: blocked_cross_mean(Xc @ R.T, Yc, chunk_size) for name, R in rotations.items()}
    differences = {name: -2*(means[a]-means[b]) for name, (a, b) in CONTRASTS.items()}
    np.testing.assert_allclose(differences["Delta_ours_true"] - differences["Delta_goicp_true"],
                               differences["Delta_ours_goicp"], atol=1e-15, rtol=0)
    if before != (benchmark.array_hash(Xc), benchmark.array_hash(Yc)) or not all(np.isfinite(v) for v in differences.values()):
        raise RuntimeError("Shared evaluation samples changed or differences are nonfinite.")
    return means, differences


def validate_n1000(population, plan, rotations, chunk_size):
    """Real 1000-point held-out pair checks; no training data or fitting."""
    Xc, Yc, details = sample_pair(population, plan, rotations["true"])
    again_X, again_Y, again_details = sample_pair(population, plan, rotations["true"])
    np.testing.assert_array_equal(Xc, again_X); np.testing.assert_array_equal(Yc, again_Y)
    assert details == again_details
    _, contrasts = evaluate_pair(Xc, Yc, rotations, chunk_size)
    full = {name: registration.mmd_squared(Xc @ R.T, Yc, SIGMA) for name, R in rotations.items()}
    cached = registration.RotationMMD(Xc, Yc, SIGMA)
    checks = []
    for name, (a, b) in CONTRASTS.items():
        direct = full[a] - full[b]
        cached_difference = cached.evaluate(rotations[a]) - cached.evaluate(rotations[b])
        checks.append({"comparison": name, "blocked_cross_only": contrasts[name],
                       "full_original_difference": direct, "absolute_discrepancy": abs(contrasts[name]-direct),
                       "cached_original_difference": cached_difference})
        np.testing.assert_allclose(contrasts[name], direct, atol=NUMERICAL_TOLERANCE, rtol=0)
        np.testing.assert_allclose(contrasts[name], cached_difference, atol=NUMERICAL_TOLERANCE, rtol=0)
    chunk_checks = []
    for chunk in sorted({257, 512, 1024, chunk_size}):
        _, values = evaluate_pair(Xc, Yc, rotations, chunk)
        discrepancy = max(abs(values[key]-contrasts[key]) for key in CONTRASTS)
        if discrepancy > NUMERICAL_TOLERANCE:
            raise RuntimeError("Chunk sizes disagree materially.")
        chunk_checks.append({"chunk_size": chunk, "max_difference_discrepancy": discrepancy})
    _, identical = evaluate_pair(Xc, Yc, {key: rotations["true"] for key in rotations}, chunk_size)
    assert all(value == 0 for value in identical.values())
    # Exact row-vector convention, using one already quantized cloud: rotating
    # by the fixed true matrix equals the target constructed from that cloud.
    probe = sampling.sample_cloud(population, np.random.default_rng(int(plan["source_sampling_seed"])), 25)
    np.testing.assert_allclose(probe @ rotations["true"].T,
                               (rotations["true"] @ probe.T).T, atol=1e-12, rtol=0)
    return {"passed": True, "evaluation_id": plan["evaluation_id"], "reproducibility_exact": True,
            "full_original_mmd_checks": checks, "chunk_size_checks": chunk_checks,
            "identical_rotation_differences_exactly_zero": True, "rotation_convention_passed": True,
            "numerical_tolerance": NUMERICAL_TOLERANCE}


def aggregate(rows):
    """Student-t intervals use whole independent resampling repetitions, not tiles."""
    summaries = []
    for n in sorted({int(row["num_points"]) for row in rows}):
        group = [row for row in rows if int(row["num_points"]) == n]
        for contrast in CONTRASTS:
            values = np.array([row[contrast] for row in group], dtype=np.float64)
            r = len(values)
            mean, std = float(values.mean()), float(values.std(ddof=1)) if r > 1 else float("nan")
            half_width = float(student_t.ppf(.975, df=r-1) * std / np.sqrt(r)) if r > 1 else float("nan")
            lower, upper = mean-half_width, mean+half_width
            summaries.append({"num_points": n, "comparison": contrast, "repetitions": r,
                              "mean_delta": mean, "sample_std_delta": std,
                              "median_delta": float(np.median(values)), "min_delta": float(values.min()),
                              "max_delta": float(values.max()), "fraction_delta_positive": float(np.mean(values > 0)),
                              "ci95_mean_lower": lower, "ci95_mean_upper": upper,
                              "ci_includes_zero": bool(lower <= 0 <= upper) if r > 1 else None,
                              "ci_method": "Student-t: mean +/- t_(r-1,0.975)*sample_std/sqrt(r)",
                              "sigma": SIGMA})
    return summaries


def plot_results(rows, summaries, training_delta, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = sorted({row["num_points"] for row in rows})
    colors = ("#2369a1", "#26845b", "#db7621")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, key, color in zip(axes, CONTRASTS, colors):
        values = [[1e6*row[key] for row in rows if row["num_points"]==n] for n in sizes]
        boxes = ax.boxplot(values, patch_artist=True, medianprops={"color":"black"})
        for box in boxes["boxes"]:
            box.set_facecolor(color); box.set_alpha(.55)
        ax.set_xticks(range(1, len(sizes)+1), [str(n) for n in sizes])
        ax.axhline(0, color="black", lw=1, label="Zero")
        if key == "Delta_ours_true":
            ax.axhline(1e6*training_delta, color="gray", ls="--", label="Training Ours - True")
            ax.legend(fontsize=8)
        ax.set(xlabel="Points per cloud", ylabel="Paired MMD difference (x 1e6)", title=LABELS[key])
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Fixed pair_017 rotations, sigma=1; independent held-out evaluations")
    fig.savefig(output/"heldout_mmd_difference_boxplot.png", dpi=180); plt.close(fig)
    ours = [row for row in summaries if row["comparison"]=="Delta_ours_true"]
    means = np.array([row["mean_delta"] for row in ours])
    errors = np.array([[row["mean_delta"]-row["ci95_mean_lower"] for row in ours],
                       [row["ci95_mean_upper"]-row["mean_delta"] for row in ours]])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(sizes, 1e6*means, yerr=1e6*errors, fmt="o-", capsize=5,
                label="Held-out mean +/- 95% Student-t CI")
    ax.axhline(0, color="black", lw=1, label="Zero")
    ax.axhline(1e6*training_delta, color="gray", ls="--", label="Training Ours - True")
    ax.set_xticks(sizes, [f"{r['num_points']}\n{r['repetitions']} repetitions" for r in ours])
    ax.set(xlabel="Points per evaluation cloud", ylabel="MMD(Ours) - MMD(True) (x 1e6)",
           title="Held-out preference for fixed rotations, sigma=1")
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.savefig(output/"heldout_mmd_difference_by_sample_size.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    all_values = [1e6*row["Delta_ours_true"] for row in rows]
    low, high = min(0., 1e6*training_delta, min(all_values)), max(0., 1e6*training_delta, max(all_values))
    padding = max(1., (high-low)*.03)
    for n, color in zip(sizes, colors):
        values = sorted(1e6*row["Delta_ours_true"] for row in rows if row["num_points"]==n)
        ax.step([low-padding]+values+[high+padding],
                [0.]+list(np.arange(1, len(values)+1)/len(values))+[1.], where="post", color=color,
                label=f"n={n}, r={len(values)}")
    ax.axvline(0, color="black", lw=1, label="Zero")
    ax.axvline(1e6*training_delta, color="gray", ls="--", label="Training Ours - True")
    ax.set(xlabel="MMD(Ours) - MMD(True) (x 1e6)", ylabel="Empirical cumulative fraction",
           title="Distribution across independent evaluation repetitions", ylim=(0,1.05))
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.savefig(output/"heldout_mmd_difference_distribution.png", dpi=180); plt.close(fig)


def report(output, rows, summaries, training_delta, validation, info, estimates, elapsed, stopped):
    ours = [r for r in summaries if r["comparison"]=="Delta_ours_true"]
    verdicts = ["True preferred" if r["ci95_mean_lower"] > 0 else "Ours preferred" if r["ci95_mean_upper"] < 0
                else "CI includes zero; not clearly distinguished" for r in ours]
    if ours and all(r["ci95_mean_lower"] > 0 for r in ours):
        conclusion = ("Held-out results support A for the evaluated sizes: fixed R_true has lower MMD than the fitted "
                      "training rotation, reversing the training preference. This supports a finite-sample estimation "
                      "error explanation for this pair. It is not a population-consistency proof.")
    elif ours and all(r["ci95_mean_upper"] < 0 for r in ours):
        conclusion = ("Held-out results support B for the evaluated sizes: the fixed Ours rotation remains preferred. "
                      "Investigate preprocessing, population geometry/symmetry and sampling conventions; this finite "
                      "diagnostic alone does not establish population-level inconsistency.")
    else:
        conclusion = ("The results are mixed or include unresolved confidence intervals (C). More independent "
                      "evaluation repetitions may help. Failure to exclude zero is not proof of equivalent rotations.")
    table = ["| n | repetitions | mean Ours - True | sample SD | 95% mean CI | fraction > 0 | interpretation |",
             "|---:|---:|---:|---:|---|---:|---|"]
    for r, verdict in zip(ours, verdicts):
        table.append(f"| {r['num_points']} | {r['repetitions']} | {r['mean_delta']:.12e} | {r['sample_std_delta']:.12e} | "
                     f"[{r['ci95_mean_lower']:.12e}, {r['ci95_mean_upper']:.12e}] | {r['fraction_delta_positive']:.3f} | {verdict} |")
    all_contrasts = ["| n | comparison | mean delta | 95% mean CI | fraction > 0 | interpretation |", "|---:|---|---:|---|---:|---|"]
    for r in summaries:
        verdict = ("second candidate preferred" if r['ci95_mean_lower'] > 0 else "first candidate preferred"
                   if r['ci95_mean_upper'] < 0 else "CI includes zero; not clearly distinguished")
        all_contrasts.append(f"| {r['num_points']} | {r['comparison']} | {r['mean_delta']:.12e} | "
                             f"[{r['ci95_mean_lower']:.12e}, {r['ci95_mean_upper']:.12e}] | {r['fraction_delta_positive']:.3f} | {verdict} |")
    stages = [{"n": n, "repetitions": len([r for r in rows if r['num_points']==n]),
               "kernel_seconds": sum(r['kernel_runtime_seconds'] for r in rows if r['num_points']==n),
               "pair_total_seconds": sum(r['runtime_seconds'] for r in rows if r['num_points']==n)} for n, _ in DESIGN]
    practical_error = max(r['absolute_discrepancy'] for r in validation['full_original_mmd_checks'])
    chunk_error = max(r['max_difference_discrepancy'] for r in validation['chunk_size_checks'])
    sd_trend = " -> ".join(f"n={r['num_points']}: {r['sample_std_delta']:.6e}" for r in ours)
    decreasing = len(ours) > 1 and all(b['sample_std_delta'] < a['sample_std_delta'] for a,b in zip(ours,ours[1:]))
    stability = ("The observed repetition-level SD decreases at each larger evaluation size, so the loss difference becomes more stable in these draws."
                 if decreasing else "The observed SD does not establish a monotone stabilization trend across these evaluation sizes.")
    widths = [r['ci95_mean_upper']-r['ci95_mean_lower'] for r in ours]
    ci_trend = ("Mean-CI widths do not shrink monotonically because larger evaluations use fewer repetitions and larger Student-t critical values."
                if any(b>a for a,b in zip(widths,widths[1:])) else "Mean-CI widths shrink in the observed evaluation stages.")
    scale_ranges = [{"n":n,"min":min(r['normalization_scale'] for r in rows if r['num_points']==n),
                     "max":max(r['normalization_scale'] for r in rows if r['num_points']==n)} for n in sorted({r['num_points'] for r in rows})]
    text = f"""# Held-out MMD validation: EMD-9515 pair_017

## Fixed training comparison and candidates

Training MMD(Ours)-MMD(True) = {training_delta:.16e}: the training objective
preferred Ours. The exact previously saved R_final, R_true and proper R_goicp
matrices are frozen; no candidate was fitted/refined and sigma stays 1.
The Go-ICP proper matrix reuses the prior explicit SO(3) projection, distance
{info['candidates']['goicp_projection_distance']:.16e}; the raw matrix and policy
are preserved in fixed_candidates.json. No transpose/inverse or translation is
used. The angular errors remain approximately 13.4114 degrees (Ours) and
0.448785 degrees (proper Go-ICP; saved raw benchmark angle about 0.449652).

## Held-out results

{chr(10).join(table)}

Positive Delta(a,b) means b has lower MMD. All three paired contrasts:

{chr(10).join(all_contrasts)}

The aggregate CSV additionally records median/min/max/sample SD. Each sample
size has its own marginal 95% Student-t interval:
mean +/- t_(r-1,0.975) * sample_SD/sqrt(r). The independent uncertainty unit is
the entire resampling repetition, not individual points, kernel pairs or tiles.
No sizes are pooled. These intervals assume independent repetitions and an
approximately normal sampling distribution of their mean; r=5 gives limited
precision. Intervals containing zero do not clearly distinguish candidates at
that size, and do not prove equivalence. The intervals are not simultaneous
multiple-comparison confidence guarantees.

## Sampling, preprocessing and exact objective

All planned seeds were stored in sampling_plan.csv before objective evaluation.
SeedSequence([master_seed, 0x48454C44, n, repetition]).spawn(2), uint64 child
integer seeds and separate default_rng streams. All 70 declared seeds are
distinct and differ from all 80 original training stream seeds. Master seed:
{info['master_seed']}. Stopping for runtime never silently changes repetitions.

Population: data/emd_9515.map, density > 0.035, {info['eligible_voxels']} voxels;
z,y,x=np.where(...), column_stack((x,y,z))*voxel_size.x. Uniform without
replacement within each independently drawn X/Z. No density weights, noise,
interpolation, origin/axis corrections, exclusions of training voxels or forced
disjointness. Coordinate values use the exact existing %.4f ASCII round-trip.
Per-repetition seeds, point counts, overlaps, XYZ/normalized-array hashes and
normalization scales are in heldout_trial_results.csv.

Y_eval=Z_eval@R_true.T; existing center_and_normalize is called once per pair.
All fixed candidates then share the same read-only Xc/Yc. Ground truth only
generates this evaluation target and defines a fixed comparison candidate; it
never chooses bandwidth, seeds, normalization or a fitted rotation.

The unchanged diagonal-excluding MMD within-cloud terms cancel under proper
rotations. Differences use -2*(mean_cross_kernel(a)-mean_cross_kernel(b)),
with exp(-squared_distance/(2*sigma^2)), sigma=1. Both arrays are tiled at
chunk_size={info['chunk_size']}; float64 tile sums plus math.fsum are exact
all-pairs evaluation up to floating-point arithmetic. No full 10000x10000
matrix, random features or approximate kernel is used. Only displayed axes
are multiplied by 1e6. Block distance algebra matches RotationMMD.problem;
direct original mmd_squared checks use the original cdist kernel.

## Numerical validation and preservation

Validation passed at n=1000 before larger stages. Maximum cross-only vs full
original MMD discrepancy: {practical_error:.16e}; maximum chunk-size difference
discrepancy across 257/512/1024 and configured size: {chunk_error:.16e}.
Saved-seed regeneration is byte/array exact, identical candidates give exact
zero difference, and row/column transformation conventions agree. Unit tests
also check shared evaluation samples, no normalization per candidate, no
fitting calls, unchanged rotation bytes and original-file hashes.

Numerical discrepancies at this scale are distinct from the across-repetition
sampling SD. CI widths estimate sampling variability, not numerical error.
protected_file_hashes.json/preservation_check.json protect pre-existing code,
frozen data and all completed numerical results. No new registrations or
sample-size scaling of fitted rotations were performed.

## Stability and computational cost

Observed Ours-vs-True sample SD by size: {sd_trend}.
{stability} {ci_trend}
Different sample sizes use independent streams; counts 20/10/5 limit how
precisely a monotone SD trend can be established. Shared normalization uses
each pair's maximum centered radius, so increasing n can also alter the scale
distribution/effective geometry; its scale range is recorded, not held fixed
or tuned. Scale ranges: {json.dumps(scale_ranges)}. Describe these results
rather than claiming a variance law. The distribution figure shows empirical
cumulative distributions, not a fitted density estimate.

CPU NumPy backend; recorded numeric diagnostic invocation elapsed
{elapsed:.3f} seconds. Per-stage times: {json.dumps(stages)}.
Pre-larger-stage quadratic estimates: {json.dumps(estimates)}.
Runtime stop reason: {stopped or 'none; all 35 predeclared repetitions completed'}.
Only tile-sized matrices (at most {info['chunk_size']}x{info['chunk_size']}) are
allocated by the blocked evaluator; multiple temporary tiles may coexist.
Software versions, environment, source hashes, plan and timing scope are in
run_metadata.json. Trial runtime includes sampling/serialization, one shared
preprocessing step and all three candidate evaluations; kernel runtime is
recorded separately. Validation, plots and I/O are excluded from trial timers.

## Interpretation and next experiment

{conclusion}

A subsequent controlled sample-size scaling study of fitted rotations is
scientifically justified to test whether training-sample error decreases with
larger independently resampled training clouds. It should use predeclared
training sizes/seeds, the unchanged sigma and algorithm, and independent
held-out evaluation. If the present intervals remain unresolved, expand
evaluation repetitions first. The present experiment freezes one training
pair's rotations and does not itself measure the effect of larger training
samples or establish performance across pairs/maps. No such registrations are
run here, and sigma was not chosen using R_true.
"""
    (output/"REPORT.md").write_text(text, encoding="utf-8")


def run(output, chunk_size=512, master_seed=DEFAULT_SEED, max_estimated_stage_seconds=600.):
    if chunk_size < 1 or not np.isfinite(max_estimated_stage_seconds) or max_estimated_stage_seconds <= 0:
        raise ValueError("Chunk size and estimated stage budget must be positive.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new/empty; no existing results will be overwritten.")
    started = perf_counter()
    rotations, candidate_info = load_rotations()
    candidate_hashes = {name: benchmark.array_hash(R) for name, R in rotations.items()}
    plan = sampling_plan(master_seed)
    pair = next(r for r in sampling.read_manifest(benchmark.DEFAULT_MANIFEST) if r['pair_id']=='pair_017')
    if sampling.sha256(sampling.DEFAULT_MAP) != pair['map_sha256']:
        raise ValueError("Original map hash mismatch.")
    population, voxel_size = sampling.load_population(sampling.DEFAULT_MAP, .035, 10000)
    if len(population) != int(pair['num_eligible_voxels']):
        raise ValueError("Threshold population changed.")
    protected = ([p for p in ROOT.glob('*.py') if p.name not in ('diagnose_emdb_heldout_mmd.py','test_emdb_heldout_mmd.py')]
                 + [sampling.DEFAULT_MAP]
                 + list(sampling.DEFAULT_OUTPUT.iterdir()) + list((ROOT/'numerical_results').rglob('*')))
    hashes = {str(p.resolve()):sampling.sha256(p) for p in protected if p.is_file()}
    training_candidates = sampling.read_manifest(DIAGNOSIS/'candidate_rotations.csv')
    training_values = {r['candidate']:float(r['mmd']) for r in training_candidates}
    training_delta = training_values['R_final']-training_values['R_true']
    output.mkdir(parents=True, exist_ok=True)
    benchmark.write_json(output/'protected_file_hashes.json', hashes)
    benchmark.write_json(output/'fixed_candidates.json', candidate_info)
    benchmark.write_checkpoint(output/'sampling_plan.csv', plan)  # BEFORE evaluation.
    info = {"design": [dict(num_points=n, repetitions=r) for n,r in DESIGN], "master_seed":master_seed,
            "seed_namespace":SEED_NAMESPACE,"sigma":SIGMA,"chunk_size":chunk_size,"backend":"CPU NumPy float64",
            "eligible_voxels":len(population),"voxel_size_x":voxel_size,"map_sha256":pair['map_sha256'],
            "candidates":candidate_info,"training_Delta_ours_true":training_delta,
            "software_versions":{name:metadata.version(name) for name in ('numpy','scipy','mrcfile','matplotlib','pymanopt','autograd')},
            "thread_environment":{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            "diagnostic_source_sha256":sampling.sha256(Path(__file__)),
            "max_estimated_stage_seconds":max_estimated_stage_seconds,"no_registrations":True,"status":"running"}
    benchmark.write_json(output/'run_metadata.json',info)
    print('Validating exact full-MMD differences, chunking, convention and deterministic held-out samples at n=1000...',flush=True)
    validation=validate_n1000(population,plan[0],rotations,chunk_size)
    benchmark.write_json(output/'numerical_validation.json',validation)
    rows,estimates,seen_hashes=[],[],set()
    stopped=None
    for n,repetitions in DESIGN:
        if n > 1000:
            first=[r for r in rows if r['num_points']==1000]
            predicted_kernel=np.median([r['kernel_runtime_seconds'] for r in first])*(n/1000)**2*repetitions
            estimate={"num_points":n,"planned_repetitions":repetitions,"predicted_kernel_seconds":float(predicted_kernel),
                      "basis":"median measured n=1000 three-candidate kernel time * (n/1000)^2 * repetitions"}
            estimates.append(estimate);benchmark.write_json(output/'runtime_estimates.json',{'stages':estimates})
            print(f'Before n={n}: projected exact kernel stage time {predicted_kernel:.2f}s for all {repetitions} repetitions.',flush=True)
            if predicted_kernel > max_estimated_stage_seconds:
                stopped=f'n={n} estimated kernel time {predicted_kernel:.2f}s exceeds predeclared practical stage budget {max_estimated_stage_seconds}s; no repetitions silently removed.'
                break
        for entry in [r for r in plan if r['num_points']==n]:
            trial_start=perf_counter()
            Xc,Yc,details=sample_pair(population,entry,rotations['true'])
            if details['X_xyz_sha256'] in seen_hashes or details['Z_xyz_sha256'] in seen_hashes:
                raise ValueError('Accidentally repeated held-out cloud; stop without resampling.')
            seen_hashes.update((details['X_xyz_sha256'],details['Z_xyz_sha256']))
            kernel_start=perf_counter()
            means,differences=evaluate_pair(Xc,Yc,rotations,chunk_size)
            kernel_time=perf_counter()-kernel_start
            row={**entry,**details,**{f'mean_cross_kernel_{name}':v for name,v in means.items()},**differences,
                 'sigma':SIGMA,'chunk_size':chunk_size,'kernel_runtime_seconds':kernel_time,
                 'runtime_seconds':perf_counter()-trial_start}
            rows.append(row)
            benchmark.write_checkpoint(output/'heldout_trial_results.csv',rows)
            print(f"{entry['evaluation_id']}: Ours-True={differences['Delta_ours_true']:+.12e}, "
                  f"scale={details['normalization_scale']:.6f}, kernel={kernel_time:.2f}s",flush=True)
        # Validate each completed stage before any larger evaluation.
        stage=[r for r in rows if r['num_points']==n]
        assert len(stage)==repetitions
        benchmark.write_checkpoint(output/'heldout_aggregated_results.csv',aggregate(rows))
        if n==1000:
            replay_X,replay_Y,replay_details=sample_pair(population,plan[0],rotations['true'])
            _,replay=evaluate_pair(replay_X,replay_Y,rotations,chunk_size)
            assert replay_details['Xc_sha256']==rows[0]['Xc_sha256'] and all(replay[k]==rows[0][k] for k in CONTRASTS)
            print('n=1000 stage completed and seed replay verified; proceeding only after the recorded runtime estimate.',flush=True)
    summaries=aggregate(rows)
    benchmark.write_checkpoint(output/'heldout_aggregated_results.csv',summaries)
    plot_results(rows,summaries,training_delta,output)
    elapsed=perf_counter()-started
    report(output,rows,summaries,training_delta,validation,info,estimates,elapsed,stopped)
    changed=[p for p,digest in hashes.items() if sampling.sha256(Path(p)) != digest]
    if changed or candidate_hashes != {name:benchmark.array_hash(R) for name,R in rotations.items()}:
        raise RuntimeError('Protected original files or frozen candidates changed.')
    benchmark.write_json(output/'preservation_check.json',{'passed':True,'protected_files':len(hashes),
                         'changed_files':changed,'fixed_candidates_unchanged':True})
    info.update(status='stopped_for_runtime' if stopped else 'completed',completed_repetitions=len(rows),
                stopping_reason=stopped,invocation_elapsed_seconds=perf_counter()-started)
    benchmark.write_json(output/'run_metadata.json',info)
    print(f"Held-out evaluation {info['status']}: {len(rows)} repetitions, {info['invocation_elapsed_seconds']:.2f}s; {output}",flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--chunk-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--max-estimated-stage-seconds',type=float,default=600.)
    args=parser.parse_args()
    run(args.output_dir.resolve(),args.chunk_size,args.seed,args.max_estimated_stage_seconds)


if __name__=='__main__':
    main()
