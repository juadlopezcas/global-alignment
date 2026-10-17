"""CPD-only convergence diagnosis for the saved, controlled 20-trial benchmark.

Fresh identity-initialized fits at caps 100, 250, 500; 1000 only for failures
at 500. Ground truth is used only after fitting. No other method is run.
The complete cap-100 reproduction gate must pass before higher caps are run.
Outputs are separate from the original benchmark. --combine reuses validated
diagnostic fits, replaces only CPD rows in a NEW directory, and never refits.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import inspect
from importlib import metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from pycpd import RigidRegistration
from pycpd.emregistration import EMRegistration

import emdb_baseline_comparison as benchmark

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "numerical_results/emdb_baseline_comparison_resumed/baseline_trial_results.csv"
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_cpd_convergence"
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def prepare_reference(reference: Path):
    """Reuse the benchmark's exact RNG, loader and shared preprocessing."""
    saved = json.loads((reference.parent / "baseline_run_metadata.json").read_text(encoding="utf-8-sig"))
    kwargs = dict(saved["config"])
    for key in ("source_cloud", "resampled_cloud"):
        kwargs[key] = Path(kwargs[key])
    config = benchmark.BaselineConfig(**kwargs)
    config.validate()
    if config.seed != 42 or config.num_trials != 20:
        raise ValueError("This controlled diagnosis requires the saved seed-42, 20-trial experiment.")
    if metadata.version("pycpd") != "2.0.0":
        raise ValueError("Inspect/review a package-version change before diagnosing this benchmark.")
    for key in THREAD_KEYS:
        if os.environ.get(key) != saved["thread_environment"].get(key):
            raise ValueError(f"Thread environment differs from the reference: {key}.")
    for key in ("numpy", "scipy", "pycpd"):
        if metadata.version(key) != saved["package_versions"][key]:
            raise ValueError(f"Reference package version mismatch: {key}.")
    X_base = benchmark.load_xyz(config.source_cloud)
    Y_base = benchmark.load_xyz(config.resampled_cloud)
    original = read_csv(reference)
    expected = [(trial, method) for trial in range(20) for method in benchmark.METHODS]
    if [(int(r["trial"]), r["method"]) for r in original] != expected:
        raise ValueError("Reference must contain exactly 20 ordered, complete four-method trials.")
    clouds, inputs = {}, []
    for trial in range(20):
        rng, _ = benchmark.trial_rngs(config.seed, trial)
        R_true = benchmark.random_rotation(3, rng)
        Xc, Yc, scale = benchmark.center_and_normalize(X_base.copy(), Y_base.copy() @ R_true.T)
        for row in original[4 * trial:4 * trial + 4]:
            if (row["emdb_id"] != benchmark.EMDB_ID or int(row["seed"]) != 42
                    or int(row["dimension"]) != 3
                    or int(row["n_source"]) != len(Xc) or int(row["n_target"]) != len(Yc)
                    or Path(row["source_file"]).resolve() != config.source_cloud.resolve()
                    or Path(row["resampled_file"]).resolve() != config.resampled_cloud.resolve()
                    or not np.array_equal(np.array(json.loads(row["R_true"])), R_true)
                    or not np.isclose(float(row["normalization_scale"]), scale, atol=1e-12, rtol=0)):
                raise ValueError(f"Reference input/RNG mismatch for trial {trial}.")
            metrics = benchmark.evaluate_rotation(R_true, np.array(json.loads(row["R_est"])), float(row["runtime_seconds"]))
            for key in benchmark.COMMON_METRICS:
                np.testing.assert_allclose(float(row[key]), metrics[key], atol=1e-10, rtol=0)
        old = original[4 * trial + 2]
        if (int(old["cpd_max_iterations"]) != 100 or float(old["cpd_tolerance"]) != 0.001
                or float(old["cpd_w"]) != 0 or old["initialization"] != "library_default_identity"):
            raise ValueError("Reference CPD settings differ from the protocol being diagnosed.")
        Xc.setflags(write=False)
        Yc.setflags(write=False)
        clouds[trial] = (Xc, Yc, R_true)
        inputs.append({"trial": trial, "Xc_sha256": array_digest(Xc), "Yc_sha256": array_digest(Yc),
                       "R_true": json.dumps(R_true.tolist()), "normalization_scale": scale})
    manifest = {
        "reference": str(reference.resolve()), "reference_sha256": digest(reference),
        "reference_metadata_sha256": digest(reference.parent / "baseline_run_metadata.json"),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "source_sha256": digest(config.source_cloud), "resampled_sha256": digest(config.resampled_cloud),
        "package_versions": {k: metadata.version(k) for k in ("numpy", "scipy", "pycpd")},
        "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
        "pycpd_em_source_sha256": digest(Path(inspect.getfile(EMRegistration))),
        "pycpd_rigid_source_sha256": digest(Path(inspect.getfile(RigidRegistration))),
        "inputs": inputs,
        "fitting": "target X=Yc, moving Y=Xc; identity R, zero t, s=1, automatic sigma2; w=0, tolerance=0.001",
        "selection": "CPD diff <= tolerance only; no ground-truth inputs to fitting or cap selection",
        "historical_input_hash_limitation": "Original metadata contains no cloud hashes. Current exact RNG/preprocessing and cap-100 reproduction validate correspondence; current input hashes are saved here.",
    }
    return original, clouds, manifest


def fit_cpd(Xc: np.ndarray, Yc: np.ndarray, cap: int):
    """Only cap differs from the original adapter; callback merely observes.

    R_true is deliberately absent. callback error=q; stopping value is diff.
    Snapshot/history disk writing occurs after the fit, outside its timer.
    """
    benchmark._validate_clouds(Xc, Yc)
    x_hash, y_hash = array_digest(Xc), array_digest(Yc)
    history = []
    previous = {"R": np.eye(3), "s": 1.0, "t": np.zeros(3), "sigma2": None}
    last_progress = perf_counter()
    start = perf_counter()
    registration = RigidRegistration(X=Yc, Y=Xc, max_iterations=cap)
    previous["sigma2"] = registration.sigma2

    def callback(iteration, error, **unused):
        nonlocal last_progress
        R = registration.R.T.copy()  # SAME conversion as run_rigid_cpd.
        t = np.asarray(registration.t).reshape(3).copy()
        history.append({
            "iteration": iteration, "cpd_internal_error": float(error), "cpd_diff": float(registration.diff),
            "sigma2": float(registration.sigma2), "scale": float(registration.s),
            "translation_norm": float(np.linalg.norm(t)), "translation": json.dumps(t.tolist()),
            "R_est": json.dumps(R.tolist()),
            "rotation_step_fro": float(np.linalg.norm(R - previous["R"], ord="fro")),
            "scale_step_abs": float(abs(registration.s - previous["s"])),
            "translation_step_norm": float(np.linalg.norm(t - previous["t"])),
            "sigma2_step_abs": float(abs(registration.sigma2 - previous["sigma2"])),
        })
        previous.update(R=R, s=float(registration.s), t=t, sigma2=float(registration.sigma2))
        if perf_counter() - last_progress >= 30:
            print(f"  iteration={iteration}, diff={registration.diff:.6g}, elapsed={perf_counter()-start:.1f}s", flush=True)
            last_progress = perf_counter()

    transformed, (scale, R_library, translation) = registration.register(callback=callback)
    runtime = perf_counter() - start
    R_est = np.asarray(R_library, dtype=float).T.copy()
    translation = np.asarray(translation, dtype=float).reshape(3)
    benchmark._check_method_rotation(R_est, "Rigid CPD diagnostic")
    finite = (float(scale), *translation, float(registration.q), float(registration.diff), float(registration.sigma2))
    if not np.isfinite(finite).all() or not np.isfinite(transformed).all():
        raise RuntimeError("Nonfinite CPD output/stopping value; do not treat it as convergence.")
    np.testing.assert_allclose(transformed, scale * (Xc @ R_est.T) + translation, atol=1e-10, rtol=1e-10)
    if (array_digest(Xc), array_digest(Yc)) != (x_hash, y_hash):
        raise RuntimeError("CPD changed its input clouds.")
    converged = bool(registration.diff <= registration.tolerance)
    row = {
        "max_iterations": cap, "actual_iterations": registration.iteration, "converged": converged,
        "stopping_criterion": "tolerance" if converged else "max_iterations",
        "final_cpd_internal_error": float(registration.q), "final_cpd_diff": float(registration.diff),
        "final_sigma2": float(registration.sigma2), "final_scale": float(scale),
        "translation_norm": float(np.linalg.norm(translation)), "translation": json.dumps(translation.tolist()),
        "runtime_seconds": runtime, "R_est": json.dumps(R_est.tolist()),
        "tolerance": registration.tolerance, "w": registration.w,
        "initialization": "library_default_identity", "Xc_sha256": x_hash, "Yc_sha256": y_hash,
    }
    return row, history


def validate_reproduction(row: dict, old: dict) -> dict:
    """Gate all higher caps on reproducing saved transforms and stop behavior."""
    R = np.array(json.loads(row["R_est"]))
    R_old = np.array(json.loads(old["R_est"]))
    t = np.array(json.loads(row["translation"]))
    t_old = np.array(json.loads(old["cpd_translation"]))
    np.testing.assert_allclose(R, R_old, atol=1e-9, rtol=0)
    np.testing.assert_allclose(t, t_old, atol=1e-9, rtol=0)
    np.testing.assert_allclose(row["final_scale"], float(old["cpd_scale"]), atol=1e-9, rtol=0)
    for key in ("rotation_error", "normalized_rotation_error", "rotation_angle_error_degrees"):
        np.testing.assert_allclose(row[key], float(old[key]), atol=1e-7, rtol=0)
    if (row["actual_iterations"] != int(old["cpd_iterations"])
            or row["stopping_criterion"] != old["cpd_stopping_criterion"]):
        raise ValueError("Cap-100 stopping behavior did not reproduce; no higher-cap fits permitted.")
    return {"trial": row["trial"], "passed": True, "rotation_difference_fro": float(np.linalg.norm(R - R_old)),
            "scale_difference_abs": abs(row["final_scale"] - float(old["cpd_scale"])),
            "translation_difference_norm": float(np.linalg.norm(t - t_old))}


def select_cap(rows: list[dict]) -> int | None:
    """Convergence-only decision: no accuracy or ground-truth fields read."""
    lookup = {(r["trial"], r["max_iterations"]): r for r in rows}
    for cap in (250, 500):
        if all(lookup[(trial, cap)]["converged"] for trial in range(20)):
            return cap
    if all(lookup[(trial, 500)]["converged"] or
           lookup[(trial, 1000)]["converged"] for trial in range(20)):
        return 1000
    return None


def summaries(rows: list[dict]) -> list[dict]:
    result = []
    for cap in sorted({r["max_iterations"] for r in rows}):
        group = [r for r in rows if r["max_iterations"] == cap]
        row = {"max_iterations": cap, "num_tested": len(group),
               "num_converged": sum(r["converged"] for r in group),
               "num_hitting_cap_without_convergence": sum(not r["converged"] for r in group),
               "num_converged_by_cap_all_20": sum(any(r["trial"] == trial and r["max_iterations"] <= cap
                                                        and r["converged"] for r in rows) for trial in range(20))}
        for key in ("actual_iterations", "runtime_seconds", "rotation_angle_error_degrees", "rotation_error",
                    "normalized_rotation_error", "final_scale", "translation_norm", "final_cpd_diff"):
            values = np.array([r[key] for r in group])
            row.update({f"{key}_mean": float(values.mean()), f"{key}_median": float(np.median(values)),
                        f"{key}_min": float(values.min()), f"{key}_max": float(values.max()),
                        f"{key}_std": float(values.std(ddof=1)) if len(values) > 1 else None})
        angles = np.array([r["rotation_angle_error_degrees"] for r in group])
        row.update(success_rate_angle_lt_5_deg=float(np.mean(angles < 5)),
                   success_rate_angle_lt_10_deg=float(np.mean(angles < 10)))
        result.append(row)
    return result


def comparisons(rows: list[dict]) -> list[dict]:
    result = []
    for trial in range(20):
        group = sorted((r for r in rows if r["trial"] == trial), key=lambda r: r["max_iterations"])
        for left, right in zip(group, group[1:]):
            result.append({"trial": trial, "from_cap": left["max_iterations"], "to_cap": right["max_iterations"],
                           "rotation_change_fro": float(np.linalg.norm(np.array(json.loads(right["R_est"])) - np.array(json.loads(left["R_est"])))),
                           "scale_change_abs": abs(right["final_scale"] - left["final_scale"]),
                           "translation_change_norm": float(np.linalg.norm(np.array(json.loads(right["translation"])) - np.array(json.loads(left["translation"])))),
                           "from_cpd_internal_error": left["final_cpd_internal_error"], "to_cpd_internal_error": right["final_cpd_internal_error"],
                           "from_cpd_diff": left["final_cpd_diff"], "to_cpd_diff": right["final_cpd_diff"],
                           "from_converged": left["converged"], "to_converged": right["converged"]})
    return result


def tail_diagnostic(trial: int, cap: int, history: list[dict]) -> dict:
    """Quantify the last 50 iterations rather than guessing stagnation."""
    tail = history[-50:]
    first, last = tail[0], tail[-1]
    q = np.array([h["cpd_internal_error"] for h in tail])
    return {"trial": trial, "max_iterations": cap, "tail_first_iteration": first["iteration"],
            "tail_last_iteration": last["iteration"], "tail_error_change": float(q[-1] - q[0]),
            "tail_error_min": float(q.min()), "tail_error_max": float(q.max()),
            "tail_num_objective_increases": int(np.sum(np.diff(q) > 0)),
            "tail_diff_min": min(h["cpd_diff"] for h in tail), "tail_diff_max": max(h["cpd_diff"] for h in tail),
            "tail_rotation_change_fro": float(np.linalg.norm(np.array(json.loads(last["R_est"])) - np.array(json.loads(first["R_est"])))),
            "tail_scale_change_abs": abs(last["scale"] - first["scale"]),
            "tail_translation_change_norm": float(np.linalg.norm(np.array(json.loads(last["translation"])) - np.array(json.loads(first["translation"])))),
            **{f"tail_max_{key}": max(h[key] for h in tail) for key in ("rotation_step_fro", "scale_step_abs", "translation_step_norm", "sigma2_step_abs")}}


def report(output: Path, rows: list[dict], gate: list[dict], tails: list[dict], selected: int | None, original: list[dict]) -> None:
    summary = summaries(rows)
    changes = comparisons(rows)
    write_csv(output / "cpd_convergence_summary.csv", summary)
    write_csv(output / "cpd_cap_comparisons.csv", changes)
    write_csv(output / "cpd_nonconverged_tail_diagnostics.csv", tails)
    selected_rows = [] if selected is None else [next(r for r in rows if r["trial"] == trial and
                      r["max_iterations"] == (500 if selected == 1000 and next(a for a in rows if a["trial"] == trial and a["max_iterations"] == 500)["converged"] else selected))
                      for trial in range(20)]
    decision = {"selected_cap": selected, "selection_uses_ground_truth": False,
                "cap_100_reproduction_passed": len(gate) == 20 and all(g["passed"] for g in gate),
                "selected_run_caps_by_trial": {r["trial"]: r["max_iterations"] for r in selected_rows}}
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# CPD convergence diagnosis: EMD-9515", "",
        "Source references: [EMRegistration.register](" + Path(inspect.getfile(EMRegistration)).as_posix() + ":130), "
        "[EMRegistration.__init__](" + Path(inspect.getfile(EMRegistration)).as_posix() + ":81), "
        "[RigidRegistration.update_variance](" + Path(inspect.getfile(RigidRegistration)).as_posix() + ":106), "
        "[RigidRegistration.update_transform](" + Path(inspect.getfile(RigidRegistration)).as_posix() + ":60), "
        "[RigidRegistration.transform_point_cloud](" + Path(inspect.getfile(RigidRegistration)).as_posix() + ":95).", "",
        "Installed pycpd 2.0.0 source was inspected and copied into installed_pycpd_sources.txt. `EMRegistration.register` iterates while `iteration < max_iterations and diff > tolerance`. The callback argument `error` is `q`, not `diff`. `RigidRegistration.update_variance` sets `diff = abs(q - qprev)`; convergence is an absolute internal-objective difference, not a variance or parameter change. Default tolerance is 0.001 and cap is 100. First q/diff are +infinity before iteration 1. NaN would also end the package loop, so this diagnostic explicitly rejects nonfinite final values.", "",
        "The objective computed using the previous variance is q = (xPx - 2*s*tr(A R) + s²*YPY)/(2*sigma²) + (D*Np/2)*log(sigma²). Variance is updated afterward. This EM auxiliary objective need not be monotone across iterations because probabilities/variance change. The package floors nonpositive updated variance to tolerance/10. The original adapter interpreted stopping correctly; tolerance was not changed.", "",
        "Target X=Yc, source Y=Xc; pycpd transforms s*Xc@R_library+t. R_est=R_library.T, so s*Xc@R_est.T+t matches the benchmark. Identity rotation, zero translation, unit scale, automatic initial variance, w=0, and scale estimation remain unchanged. Ground truth enters only post-fit evaluation; cap selection reads only convergence flags. Every cap uses a fresh fit from the same initialization, not a warm start.", "",
        f"Cap-100 reproduction: {len(gate)}/20 passed; rotation/scale/translation atol 1e-9, accuracy metrics atol 1e-7, exact iteration/stop criterion. Runtime is not expected to reproduce. Maximum rotation discrepancy: {max(g['rotation_difference_fro'] for g in gate):.6g}.", "",
        "Current Xc/Yc hashes, cloud file hashes, exact R_true, versions, thread environment and original CSV hash are in input_manifest.json. Exact shared preprocessing and RNG functions are reused from the benchmark. Original metadata has no historical cloud hashes; reproduction of all 20 cap-100 results supplies the retrospective input consistency check. Read-only inputs and before/after hashes verify fits do not mutate data.", "",
        "Fit timers include setup/fitting and lightweight callback snapshots, exclude preprocessing/evaluation/history disk writes. Callback overhead is included, so these diagnostic runtimes are not strictly identical to uninstrumented baseline timings.", "",
        "## Convergence (cap 1000 tests only failures at 500)", "",
        "| Cap | Tested | Converged | Cap hit without convergence | Iterations mean / median / range | Runtime mean / median (s) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for s in summary:
        lines.append(f"| {s['max_iterations']} | {s['num_tested']} | {s['num_converged']} | {s['num_hitting_cap_without_convergence']} | {s['actual_iterations_mean']:.2f} / {s['actual_iterations_median']:.1f} / {s['actual_iterations_min']}–{s['actual_iterations_max']} | {s['runtime_seconds_mean']:.3f} / {s['runtime_seconds_median']:.3f} |")
    lines += ["", f"Selected final cap, solely by convergence: **{selected}**. " + ("All trials satisfy unchanged tolerance." if selected else "No sufficient cap established through 1000. Do not update the benchmark or loosen tolerance automatically; review the last-50-iteration diagnostics."), "",
              "## Ground-truth metrics (interpretation only)", "",
              "| Cap | Angular mean / median / min / max (deg) | Success <5° / <10° | Scale mean / median / range |", "|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['max_iterations']} | {s['rotation_angle_error_degrees_mean']:.6f} / {s['rotation_angle_error_degrees_median']:.6f} / {s['rotation_angle_error_degrees_min']:.6f} / {s['rotation_angle_error_degrees_max']:.6f} | {s['success_rate_angle_lt_5_deg']:.1%} / {s['success_rate_angle_lt_10_deg']:.1%} | {s['final_scale_mean']:.6f} / {s['final_scale_median']:.6f} / {s['final_scale_min']:.6f}–{s['final_scale_max']:.6f} |")
    for cap in sorted({r["to_cap"] for r in changes}):
        group = [r for r in changes if r["to_cap"] == cap]
        values = np.array([r["rotation_change_fro"] for r in group])
        lines += ["", f"Rotation change into cap {cap}: mean {values.mean():.6g}, median {np.median(values):.6g}, max {values.max():.6g}. Full per-trial rotation, scale, translation and internal-error changes are in cpd_cap_comparisons.csv."]
    if selected_rows:
        iterations = np.array([r["actual_iterations"] for r in selected_rows])
        final_changes = []
        for final in selected_rows:
            initial = next(r for r in rows if r["trial"] == final["trial"] and r["max_iterations"] == 100)
            final_changes.append({
                "trial": final["trial"], "selected_cap": selected, "actually_tested_cap": final["max_iterations"],
                "rotation_change_from_100_fro": float(np.linalg.norm(np.array(json.loads(final["R_est"])) - np.array(json.loads(initial["R_est"])))),
                "scale_change_from_100_abs": abs(final["final_scale"] - initial["final_scale"]),
                "translation_change_from_100_norm": float(np.linalg.norm(np.array(json.loads(final["translation"])) - np.array(json.loads(initial["translation"])))),
            })
        write_csv(output / "cpd_100_to_convergence_changes.csv", final_changes)
        final_summary = summaries([{**r, "max_iterations": selected} for r in selected_rows])[0]
        write_csv(output / "cpd_selected_results_summary.csv", [final_summary])
        rotation_changes = np.array([r["rotation_change_from_100_fro"] for r in final_changes])
        lines += ["", f"Total rotation change from cap 100 to convergence: mean {rotation_changes.mean():.8g}, median {np.median(rotation_changes):.8g}, min {rotation_changes.min():.8g}, max {rotation_changes.max():.8g}. Per-trial changes are in cpd_100_to_convergence_changes.csv."]
        lines += ["", f"Actual iterations to convergence: mean {iterations.mean():.2f}, median {np.median(iterations):.1f}, min {iterations.min()}, max {iterations.max()}; per trial {iterations.tolist()}.", "",
                  "## Old truncated vs converged CPD", "", "| Quantity | Old cap 100 | Selected converged results |", "|---|---:|---:|"]
        for key in benchmark.COMMON_METRICS:
            oldvals = np.array([float(r[key]) for r in original if r["method"] == "cpd"])
            vals = np.array([r[key] for r in selected_rows])
            lines.append(f"| {key} mean | {oldvals.mean():.8f} | {vals.mean():.8f} |")
        angles = np.array([r["rotation_angle_error_degrees"] for r in selected_rows])
        lines += ["", f"Large-angle failures remain: {int(np.sum(angles > 90))}/20 above 90 degrees. These converged local failures are retained without restart, GT initialization, sign changes or convention changes."]
        lines += ["", f"All-20 converged angular error: mean {angles.mean():.8f}, sample SD {angles.std(ddof=1):.8f}, median {np.median(angles):.8f}, range {angles.min():.8f}–{angles.max():.8f} degrees; success <5 degrees {np.mean(angles < 5):.1%}, <10 degrees {np.mean(angles < 10):.1%}.",
                  "", f"All-20 converged scale: mean {final_summary['final_scale_mean']:.8f}, sample SD {final_summary['final_scale_std']:.8f}, median {final_summary['final_scale_median']:.8f}, range {final_summary['final_scale_min']:.8f}–{final_summary['final_scale_max']:.8f}. Scale estimation was retained even for large-angle local failures."]
        if selected == 1000:
            lines += ["", "The cap-1000 experiment tests only failures at 500. All other replacement rows reuse their converged cap-500 fits: the installed loop would stop at exactly the same iteration with a 1000 cap. The combined CSV records the selected cap as cpd_max_iterations and the actually tested cap separately as cpd_diagnostic_tested_cap. No additional CPD fits are needed for already converged trials."]
    if tails:
        lines += ["", "## Nonconverged tail inspection", "", "Each row below summarizes last 50 iterations. q is an EM auxiliary objective; change does not by itself certify progress/convergence. No tolerance changes were made.", ""]
        for t in tails:
            lines.append(f"- Trial {t['trial']}, cap {t['max_iterations']}: q change {t['tail_error_change']:.8g}, diff range {t['tail_diff_min']:.8g}–{t['tail_diff_max']:.8g}, rotation change {t['tail_rotation_change_fro']:.8g}, scale change {t['tail_scale_change_abs']:.8g}, translation change {t['tail_translation_change_norm']:.8g}.")
            h = read_csv(output / "histories" / f"trial_{int(t['trial']):02d}_cap_{int(t['max_iterations'])}.csv")
            last = h[-1]
            lines += [f"  q increased on {int(t['tail_num_objective_increases'])} of the last {len(h[-50:]) - 1} transitions, so it was not monotonically decreasing. Last five absolute objective differences: {[float(r['cpd_diff']) for r in h[-5:]]}. Final per-iteration changes: rotation {float(last['rotation_step_fro']):.8g}, scale {float(last['scale_step_abs']):.8g}, translation {float(last['translation_step_norm']):.8g}, variance {float(last['sigma2_step_abs']):.8g}."]
            final = next((r for r in selected_rows if r["trial"] == t["trial"]), None)
            if final:
                lines += [f"  The same identity-initialized trial subsequently met the original tolerance at iteration {final['actual_iterations']} (diff {final['final_cpd_diff']:.8g}). The cap-{t['max_iterations']} tail therefore represented slow continued convergence, not a persistent numerical stagnation requiring a tolerance change."]
    lines += ["", f"Total measured CPD fitting time across all tested caps: {sum(r['runtime_seconds'] for r in rows):.3f} seconds.", "",
              "Original benchmark remains untouched. `--combine` validates the reproduction gate, input manifest, selected fits, finite transforms, SO(3), scale/translation and all original non-CPD fields, then writes replacements and combined results into a new directory and calls existing aggregation/plotting only. No Ours, ICP or Go-ICP fit is invoked.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(reference: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Diagnostic output must be empty; existing results will not be overwritten.")
    original, clouds, manifest = prepare_reference(reference)
    output.mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir()
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "installed_pycpd_sources.txt").write_text(
        f"pycpd {metadata.version('pycpd')}\n{inspect.getfile(EMRegistration)}\n" + inspect.getsource(EMRegistration) +
        f"\n{inspect.getfile(RigidRegistration)}\n" + inspect.getsource(RigidRegistration), encoding="utf-8")
    rows, gate, tails = [], [], []
    started = perf_counter()
    for cap in (100, 250, 500, 1000):
        trials = range(20) if cap != 1000 else [r["trial"] for r in rows if r["max_iterations"] == 500 and not r["converged"]]
        for trial in trials:
            print(f"CPD trial {trial+1}/20, cap={cap}", flush=True)
            Xc, Yc, R_true = clouds[trial]
            fitted, history = fit_cpd(Xc, Yc, cap)
            row = {"trial": trial, "seed": 42, "emdb_id": benchmark.EMDB_ID, **fitted,
                   "R_true": json.dumps(R_true.tolist()),
                   **benchmark.evaluate_rotation(R_true, np.array(json.loads(fitted["R_est"])), fitted["runtime_seconds"])}
            write_csv(output / "histories" / f"trial_{trial:02d}_cap_{cap}.csv", history)
            rows.append(row)
            write_csv(output / "cpd_convergence_trials.csv", rows)
            if cap == 100:
                gate.append(validate_reproduction(row, original[4 * trial + 2]))
                write_csv(output / "cpd_reproduction_check.csv", gate)
            if cap >= 500 and not row["converged"]:
                tails.append(tail_diagnostic(trial, cap, history))
            print(f"  iterations={row['actual_iterations']}, {row['stopping_criterion']}, diff={row['final_cpd_diff']:.6g}, runtime={row['runtime_seconds']:.2f}s", flush=True)
        if cap == 100:
            if len(gate) != 20:
                raise RuntimeError("Incomplete reproduction gate; higher caps forbidden.")
            print("All 20 original cap-100 transforms reproduced; higher-cap tests authorized by gate.", flush=True)
    selected = select_cap(rows)
    report(output, rows, gate, tails, selected, original)
    print(f"Finished CPD-only diagnosis in {perf_counter()-started:.2f}s; convergence-selected cap={selected}.", flush=True)


def parsed_diagnostics(path: Path) -> list[dict]:
    rows = read_csv(path)
    for row in rows:
        for key in ("trial", "seed", "max_iterations", "actual_iterations"):
            row[key] = int(row[key])
        row["converged"] = row["converged"] == "True"
        for key in ("final_cpd_internal_error", "final_cpd_diff", "final_sigma2", "final_scale", "translation_norm",
                    "tolerance", "w", *benchmark.COMMON_METRICS):
            row[key] = float(row[key])
    return rows


def combine(reference: Path, output: Path, combined_output: Path) -> None:
    """No fitting: replace ONLY validated CPD rows, preserve original others."""
    if combined_output.exists() and any(combined_output.iterdir()):
        raise ValueError("Combined destination must be new/empty; no benchmark will be overwritten.")
    if combined_output.resolve() in (reference.parent.resolve(), output.resolve()):
        raise ValueError("Combined output must be separate from reference and diagnostic outputs.")
    original, clouds, manifest = prepare_reference(reference)
    saved_manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    if manifest != saved_manifest:
        raise ValueError("Input/reference/environment manifest changed; replacement refused.")
    rows = parsed_diagnostics(output / "cpd_convergence_trials.csv")
    lookup = {(r["trial"], r["max_iterations"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate diagnostic trial/cap rows.")
    gate = [validate_reproduction(lookup[(trial, 100)], original[trial * 4 + 2]) for trial in range(20)]
    cap = select_cap(rows)
    if cap is None:
        raise ValueError("No sufficient convergence-only cap through 1000; replacement refused.")
    tails_path = output / "cpd_nonconverged_tail_diagnostics.csv"
    tails = []
    if tails_path.exists():
        tails = [{k: int(v) if k in ("trial", "max_iterations", "tail_first_iteration", "tail_last_iteration", "tail_num_objective_increases")
                  else float(v) for k, v in row.items()} for row in read_csv(tails_path)]
    report(output, rows, gate, tails, cap, original)
    combined, replacements = [], []
    for old in original:
        trial = int(old["trial"])
        if old["method"] != "cpd":
            combined.append(dict(old))
            continue
        run_cap = 500 if cap == 1000 and lookup[(trial, 500)]["converged"] else cap
        row = lookup[(trial, run_cap)]
        Xc, Yc, R_true = clouds[trial]
        if (row["R_true"] != old["R_true"] or row["Xc_sha256"] != array_digest(Xc)
                or row["Yc_sha256"] != array_digest(Yc) or row["tolerance"] != 0.001 or row["w"] != 0
                or not row["converged"] or row["final_cpd_diff"] > row["tolerance"]
                or row["initialization"] != "library_default_identity"):
            raise ValueError(f"Replacement protocol/inputs/convergence mismatch for trial {trial}.")
        R_est = np.array(json.loads(row["R_est"]))
        metrics = benchmark.evaluate_rotation(R_true, R_est, row["runtime_seconds"])
        for key in benchmark.COMMON_METRICS:
            np.testing.assert_allclose(metrics[key], row[key], atol=1e-10, rtol=0)
        t = np.array(json.loads(row["translation"]))
        if t.shape != (3,) or not np.isfinite(t).all() or not np.isfinite(row["final_scale"]) or row["final_scale"] <= 0:
            raise ValueError("Invalid replacement scale/translation.")
        np.testing.assert_allclose(np.linalg.norm(t), row["translation_norm"], atol=1e-12, rtol=0)
        new = dict(old)
        new.update(R_est=row["R_est"], **metrics, cpd_scale=row["final_scale"],
                   cpd_translation=row["translation"], cpd_translation_norm=row["translation_norm"],
                   cpd_iterations=row["actual_iterations"], cpd_max_iterations=cap,
                   cpd_stopping_criterion="tolerance")
        # A cap-500 converged trajectory is identical with cap=1000 because the
        # loop stops on tolerance first; retain its actual tested cap explicitly.
        new.update(cpd_diagnostic_tested_cap=run_cap, cpd_converged=True,
                   cpd_final_internal_error=row["final_cpd_internal_error"], cpd_final_diff=row["final_cpd_diff"],
                   cpd_final_sigma2=row["final_sigma2"])
        replacements.append(new)
        combined.append(new)
    fields = list(dict.fromkeys(k for row in combined for k in row))
    combined = [{k: r.get(k, "") for k in fields} for r in combined]
    for old, new in zip(original, combined):
        if old["method"] != "cpd" and any(old[k] != new[k] for k in old):
            raise RuntimeError("A non-CPD field changed; replacement refused.")
    combined_output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cpd_replacement_results.csv", replacements)
    write_csv(combined_output / "baseline_trial_results.csv", combined)
    reread = read_csv(combined_output / "baseline_trial_results.csv")
    for old, new in zip(original, reread):
        if old["method"] != "cpd" and any(old[k] != new[k] for k in old):
            raise RuntimeError("Serialized non-CPD values changed.")
    numeric = [dict(r, trial=int(r["trial"]), **{key: float(r[key]) for key in benchmark.COMMON_METRICS}) for r in reread]
    write_csv(combined_output / "baseline_aggregated_results.csv", benchmark.aggregate_results(numeric))
    benchmark.save_plots(numeric, combined_output)
    meta = json.loads((reference.parent / "baseline_run_metadata.json").read_text(encoding="utf-8-sig"))
    meta["cpd_convergence_update"] = {"diagnostic_directory": str(output.resolve()), "selected_cap": cap,
        "selection": "all 20 satisfy unchanged pycpd objective-difference tolerance; no accuracy tuning",
        "reference_csv": str(reference.resolve()), "reference_sha256": digest(reference),
        "preserved_non_cpd_rows": 60, "replaced_cpd_rows": 20,
        "no_non_cpd_fits_run": True, "cpd_runtime_scope": "setup/fitting including observational callback, excluding history disk writes"}
    (combined_output / "baseline_run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    validation = {"reference_csv_unchanged": digest(reference) == manifest["reference_sha256"],
                  "non_cpd_rows_preserved_field_for_field": True, "same_twenty_trials_and_ground_truth": True,
                  "same_cloud_hashes": True, "rotations_finite_SO3": True, "scale_translation_finite_recorded": True,
                  "no_ground_truth_in_fitting_or_selection": True, "selected_cap": cap,
                  "combined_output": str(combined_output.resolve())}
    (output / "replacement_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    with (output / "REPORT.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\nValidated CPD-only replacements and recomputed aggregates/plots written to `{combined_output.resolve()}`. All 60 non-CPD rows preserve every original CSV field exactly. Original CSV/hash unchanged. `cpd_replacement_results.csv` contains 20 validated replacement rows.\n")
    print(json.dumps(validation, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--combine", action="store_true", help="validate and combine saved fits only; no refitting")
    parser.add_argument("--combined-output-dir", type=Path,
                        default=ROOT / "numerical_results/emdb_baseline_comparison_cpd_converged")
    args = parser.parse_args()
    if args.combine:
        combine(args.reference_csv, args.output_dir, args.combined_output_dir)
    else:
        run(args.reference_csv, args.output_dir)


if __name__ == "__main__":
    main()
