"""Controlled comparison on frozen independent EMDB resampling pairs.

Standalone entry point; reuse the validated adapters/core from the preserved
emdb_baseline_comparison.py without changing them. No on-the-fly sampling or
rotation generation. Y=Z@R_true.T; evaluation measures recovery of the imposed
population rotation, NOT exact pointwise alignment between finite resamples.

Default is a two-pair smoke test. --num-pairs 20 must be explicit. Per-method
Atomic per-method CSV checkpoints and --resume-from preserve completed fits.
--preflight-only validates inputs/settings without running registrations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

import emdb_baseline_comparison as registration
import generate_emdb_resampling_pairs as dataset
import emdb_resampling_diagnostics as geometry

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = dataset.DEFAULT_OUTPUT / "resampling_manifest.csv"
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_independent_resampling"
DEFAULT_CONVENTION = ROOT / "numerical_results/emdb_goicp_feasibility/final_convention_status.json"
CONFIG = registration.BaselineConfig()
EXTRA_FIELDS = ("cpd_converged", "goicp_translation_boundary_margin", "goicp_translation_near_boundary",
                "goicp_translation_boundary_diagnostic_tolerance", "goicp_translation_x",
                "goicp_translation_y", "goicp_translation_z", "goicp_translation_near_boundary_axes")
LEGACY_DRIVER_SHA256 = "caf134b9215bb6bb2073b6c35e1f04e60d8e833915006d3c17a81d5fa35092de"


def array_hash(array: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def prepare_pair(row: dict, manifest: Path):
    """Load frozen quantized clouds/rotation; preprocess exactly once."""
    X = registration.load_xyz(dataset.resolve_cloud(manifest, row["source_file"]))
    Z = registration.load_xyz(dataset.resolve_cloud(manifest, row["target_unrotated_file"]))
    R_true = np.array(json.loads(row["R_true"]), dtype=float)
    registration.assert_rotation(R_true)
    Y = Z @ R_true.T
    Xc, Yc, normalization_scale = registration.center_and_normalize(X, Y)
    Xc.setflags(write=False)
    Yc.setflags(write=False)
    R_true.setflags(write=False)
    return Xc, Yc, R_true, normalization_scale


def iter_pair_results(row: dict, manifest: Path, *, completed_methods=()):
    """Shared array objects for every adapter; ground truth only in evaluation."""
    Xc, Yc, R_true, scale = prepare_pair(row, manifest)
    hashes = (array_hash(Xc), array_hash(Yc), array_hash(R_true))
    geometry_diagnostics = geometry.pair_diagnostics(Xc, Yc)
    fit_rng = np.random.default_rng(int(row["ours_fit_seed"]))
    runners = (
        ("ours", lambda: registration.run_ours(Xc, Yc, fit_rng, CONFIG)),
        ("icp", lambda: registration.run_icp(Xc, Yc, max_correspondence_distance=CONFIG.icp_max_correspondence_distance)),
        ("cpd", lambda: registration.run_rigid_cpd(Xc, Yc)),
        ("goicp", lambda: registration.run_goicp(
            Xc, Yc, mse_threshold=CONFIG.goicp_mse_threshold, dt_size=CONFIG.goicp_dt_size,
            trim_fraction=CONFIG.goicp_trim_fraction, num_data_points=CONFIG.goicp_num_data_points,
            translation_min=CONFIG.goicp_translation_min, translation_width=CONFIG.goicp_translation_width)),
    )
    for method, fit in runners:
        if method in completed_methods:
            continue
        print(f"Starting {row['pair_id']} {method} with frozen, shared 1000-point inputs.", flush=True)
        result = fit()  # No R_true argument to any registration adapter.
        if (array_hash(Xc), array_hash(Yc), array_hash(R_true)) != hashes:
            raise RuntimeError("A method changed the shared inputs/ground truth.")
        output = {
            "emdb_id": row["emdb_id"], "pair_id": row["pair_id"], "method": method,
            "map_file": str(Path(row["map_file"]).resolve()), "map_sha256": row["map_sha256"],
            "contour_threshold": dataset.contour_threshold(row),
            "dimension": 3, "master_seed": int(row["master_seed"]),
            "source_file": str(dataset.resolve_cloud(manifest, row["source_file"])),
            "target_unrotated_file": str(dataset.resolve_cloud(manifest, row["target_unrotated_file"])),
            **{name: int(row[name]) for name in dataset.STREAM_NAMES},
            "source_sha256": row["source_sha256"], "target_sha256": row["target_sha256"],
            "Xc_sha256": hashes[0], "Yc_sha256": hashes[1],
            "R_true": row["R_true"], "R_est": json.dumps(result.R_est.tolist()),
            "n_source": len(Xc), "n_target": len(Yc), "normalization_scale": scale,
            **geometry_diagnostics,
            **registration.evaluate_rotation(R_true, result.R_est, result.runtime_seconds),
            "sigma": CONFIG.sigma, "max_greedy_moves": CONFIG.max_greedy_moves,
            "greedy_tolerance": CONFIG.greedy_tolerance, "max_iterations": CONFIG.max_iterations,
            "pca_max_iterations": CONFIG.pca_max_iterations, "min_gradient_norm": CONFIG.min_gradient_norm,
            **{field: None for field in (*registration.DIAGNOSTIC_FIELDS, *EXTRA_FIELDS)},
        }
        output.update(result.diagnostics)
        if method == "ours":
            error, normalized = registration.signed_permutation_frame_error(
                R_true @ result.registration.U_X, result.registration.U_Y)
            output.update(frame_error=error, normalized_frame_error=normalized)
        if method == "cpd":
            if (output["cpd_max_iterations"] != 1000 or output["cpd_tolerance"] != 0.001 or output["cpd_w"] != 0):
                raise RuntimeError("CPD no longer matches its validated configuration.")
            output["cpd_converged"] = output["cpd_stopping_criterion"] == "tolerance"
            if not output["cpd_converged"] and output["cpd_iterations"] != 1000:
                raise RuntimeError("CPD exited before its cap without satisfying tolerance; investigate numerical failure.")
        if method == "goicp":
            t = np.array(json.loads(output["goicp_translation"]))
            lower = CONFIG.goicp_translation_min
            upper = lower + CONFIG.goicp_translation_width
            margin = float(np.minimum(t - lower, upper - t).min())
            # Fixed post-fit diagnostic (0.1% of the cube width); never tuning.
            boundary_tolerance = 1e-4
            output.update(goicp_translation_boundary_margin=margin,
                          goicp_translation_near_boundary=margin <= boundary_tolerance,
                          goicp_translation_boundary_diagnostic_tolerance=boundary_tolerance,
                          goicp_translation_x=float(t[0]), goicp_translation_y=float(t[1]),
                          goicp_translation_z=float(t[2]),
                          goicp_translation_near_boundary_axes=json.dumps(
                              (np.minimum(t-lower, upper-t) <= boundary_tolerance).tolist()))
        for field, value in output.items():
            if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise RuntimeError(f"Nonfinite diagnostic: {row['pair_id']} {method} {field}.")
        yield output


def method_settings() -> dict:
    settings = {k: v for k, v in asdict(CONFIG).items()
                if k not in ("source_cloud", "resampled_cloud", "num_trials", "seed")}
    settings.update(cpd_max_iterations=1000, cpd_tolerance=0.001, cpd_w=0,
                    cpd_initialization="identity R, t=0, s=1, automatic sigma2; scale estimated",
                    goicp_dt_expand_factor=2.0, goicp_cleanup_guard=registration.GOICP_ROUNDOFF_LIMIT)
    return settings


def validate_saved_rows(rows: list[dict], pairs: list[dict], manifest: Path, num_pairs: int,
                        *, allow_legacy: bool = False) -> None:
    expected = [(p["pair_id"], method) for p in pairs[:num_pairs] for method in registration.METHODS]
    if [(r["pair_id"], r["method"]) for r in rows] != expected[:len(rows)] or len(rows) > len(expected):
        raise ValueError("Saved results must be an ordered pair/method prefix with no duplicates.")
    lookup = {p["pair_id"]: p for p in pairs}
    prepared = {}
    for r in rows:
        p = lookup[r["pair_id"]]
        if p["pair_id"] not in prepared:
            prepared[p["pair_id"]] = prepare_pair(p, manifest)
        Xc, Yc, R_true, scale = prepared[p["pair_id"]]
        if r["emdb_id"] != p["emdb_id"] or int(r["master_seed"]) != int(p["master_seed"]):
            raise ValueError("Saved dataset identity/seed mismatch.")
        if not (allow_legacy and p["emdb_id"] == "EMD-9515" and "map_file" not in r):
            if (r.get("map_file") != str(Path(p["map_file"]).resolve())
                    or r.get("map_sha256") != p["map_sha256"]
                    or float(r.get("contour_threshold", "nan")) != dataset.contour_threshold(p)):
                raise ValueError("Saved map identity/contour threshold mismatch.")
        for key in ("R_true", "source_sha256", "target_sha256"):
            if r[key] != p[key]:
                raise ValueError(f"Frozen dataset mismatch: {key}.")
        for key in dataset.STREAM_NAMES:
            if int(r[key]) != int(p[key]):
                raise ValueError(f"Saved seed mismatch: {key}.")
        if (r["Xc_sha256"] != array_hash(Xc) or r["Yc_sha256"] != array_hash(Yc)
                or int(r["n_source"]) != 1000 or int(r["n_target"]) != 1000
                or Path(r["source_file"]) != dataset.resolve_cloud(manifest, p["source_file"])
                or Path(r["target_unrotated_file"]) != dataset.resolve_cloud(manifest, p["target_unrotated_file"])):
            raise ValueError("Saved input files/arrays do not match frozen pair.")
        np.testing.assert_allclose(float(r["normalization_scale"]), scale, atol=1e-12, rtol=0)
        metrics = registration.evaluate_rotation(R_true, np.array(json.loads(r["R_est"])), float(r["runtime_seconds"]))
        for key in registration.COMMON_METRICS:
            np.testing.assert_allclose(float(r[key]), metrics[key], atol=1e-10, rtol=0)
        for key in ("sigma", "max_greedy_moves", "greedy_tolerance", "max_iterations",
                    "pca_max_iterations", "min_gradient_norm"):
            if float(r[key]) != getattr(CONFIG, key):
                raise ValueError(f"Saved Ours settings mismatch: {key}.")
        if r["method"] == "icp" and float(r["icp_max_correspondence_distance"]) != CONFIG.icp_max_correspondence_distance:
            raise ValueError("Saved ICP settings mismatch.")
        if r["method"] == "cpd" and (int(r["cpd_max_iterations"]) != 1000
                or float(r["cpd_tolerance"]) != .001 or float(r["cpd_w"]) != 0):
            raise ValueError("Saved CPD settings mismatch.")
        if r["method"] == "cpd":
            converged = r["cpd_stopping_criterion"] == "tolerance"
            if is_true(r["cpd_converged"]) != converged or not 1 <= int(r["cpd_iterations"]) <= 1000:
                raise ValueError("Inconsistent CPD convergence diagnostics.")
            if not converged and int(r["cpd_iterations"]) != 1000:
                raise ValueError("CPD stopped early without tolerance convergence.")
        if r["method"] == "goicp":
            for key, value in (("goicp_dt_size", 300), ("goicp_num_data_points", 1000),
                               ("goicp_mse_threshold", .001), ("goicp_trim_fraction", 0),
                               ("goicp_translation_min", -.05), ("goicp_translation_width", .10),
                               ("goicp_dt_expand_factor", 2), ("goicp_cleanup_tolerance", 5e-6)):
                if float(r[key]) != value:
                    raise ValueError(f"Saved Go-ICP settings mismatch: {key}.")
            cleaned, diagnostics = registration.clean_goicp_rotation(json.loads(r["goicp_raw_rotation"]))
            np.testing.assert_allclose(cleaned, np.array(json.loads(r["R_est"])), atol=1e-12, rtol=0)
            if is_true(r["goicp_projection_applied"]) != diagnostics["goicp_projection_applied"]:
                raise ValueError("Saved Go-ICP projection flag is inconsistent with its raw rotation.")
            np.testing.assert_allclose(float(r["goicp_projection_distance"]), diagnostics["goicp_projection_distance"], atol=1e-12, rtol=0)
            t = np.array(json.loads(r["goicp_translation"]))
            if t.shape != (3,) or not np.isfinite(t).all():
                raise ValueError("Invalid saved Go-ICP translation.")
            margin = float(np.minimum(t + .05, .05 - t).min())
            np.testing.assert_allclose(float(r["goicp_translation_boundary_margin"]), margin, atol=1e-12, rtol=0)
            if is_true(r["goicp_translation_near_boundary"]) != (margin <= 1e-4):
                raise ValueError("Saved translation-boundary flag is inconsistent.")


def pairwise_table(rows: list[dict]) -> list[dict]:
    results = []
    for pair_id in dict.fromkeys(r["pair_id"] for r in rows):
        group = {r["method"]: r for r in rows if r["pair_id"] == pair_id}
        if set(group) != set(registration.METHODS):
            raise ValueError("Pairwise table requires all four methods per pair.")
        record = {"pair_id": pair_id}
        for method in registration.METHODS:
            record[f"{method}_angle"] = float(group[method]["rotation_angle_error_degrees"])
            record[f"{method}_runtime_seconds"] = float(group[method]["runtime_seconds"])
        results.append(record)
    return results


def save_plots(rows: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ours": "#2369a1", "icp": "#db7621", "cpd": "#26845b", "goicp": "#8751a1"}
    title = f"{rows[0]['emdb_id']}: {len(rows)//4} independent resampling pairs"
    for metric, ylabel, filename in (
        ("rotation_angle_error_degrees", "Imposed-rotation error (degrees)", "independent-pairs-angle-error-boxplot.png"),
        ("runtime_seconds", "Runtime (seconds)", "independent-pairs-runtime-boxplot.png")):
        values = [[float(r[metric]) for r in rows if r["method"] == method] for method in registration.METHODS]
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        boxes = ax.boxplot(values, patch_artist=True, medianprops={"color": "black"})
        for box, method in zip(boxes["boxes"], registration.METHODS):
            box.set_facecolor(colors[method]); box.set_alpha(.65)
        ax.set_xticks(range(1, 5), [registration.METHOD_LABELS[m] for m in registration.METHODS])
        ax.set(ylabel=ylabel, title=title, ylim=(0, max(1, max(max(g) for g in values)*1.1)))
        ax.grid(axis="y", alpha=.25)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
    table = pairwise_table(rows)
    fig, ax = plt.subplots(figsize=(max(7, len(table)*.45), 4.5), constrained_layout=True)
    for method in registration.METHODS:
        ax.plot(range(len(table)), [r[f"{method}_angle"] for r in table], "o-",
                color=colors[method], label=registration.METHOD_LABELS[method])
    ax.set_xticks(range(len(table)), [r["pair_id"] for r in table], rotation=45)
    ax.set(xlabel="Frozen resampling pair", ylabel="Imposed-rotation error (degrees)", title=title,
           ylim=(0, max(1, max(float(r['rotation_angle_error_degrees']) for r in rows)*1.1)))
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(output_dir / "independent-pairs-angle-error-by-pair.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
    x, y = [r["ours_angle"] for r in table], [r["goicp_angle"] for r in table]
    limit = max(1, max(x+y)*1.15)
    ax.plot([0, limit], [0, limit], "--", color="gray", label="y = x")
    ax.scatter(x, y, color=colors["ours"])
    if len(table) <= 5:
        for row in table:
            ax.annotate(row["pair_id"], (row["ours_angle"], row["goicp_angle"]), xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="Ours imposed-rotation error (degrees)", ylabel="Go-ICP imposed-rotation error (degrees)",
           title=title, xlim=(0, limit), ylim=(0, limit), aspect="equal")
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(output_dir / "ours-vs-goicp-angle-scatter.png", dpi=180)
    plt.close(fig)


def is_true(value) -> bool:
    return value is True or value == "True"


def smoke_status(rows: list[dict]) -> dict:
    cpd = [r for r in rows if r["method"] == "cpd"]
    goicp = [r for r in rows if r["method"] == "goicp"]
    return {"num_pairs": len(rows)//4, "num_rows": len(rows), "all_trial_metrics_finite": True,
            "all_rotations_valid_SO3": True, "shared_inputs_and_frozen_rotations_validated": True,
            "cpd_all_converged": all(is_true(r["cpd_converged"]) for r in cpd),
            "goicp_all_completed": len(goicp) == len(rows)//4,
            "goicp_any_translation_near_boundary": any(is_true(r["goicp_translation_near_boundary"]) for r in goicp),
            "goicp_projection_count": sum(is_true(r["goicp_projection_applied"]) for r in goicp)}


def preflight(manifest: Path, num_pairs: int,
              convention_record: Path = DEFAULT_CONVENTION) -> tuple[list[dict], dict, dict]:
    """Validate all frozen data and imports, reusing the saved convention check.

    No method adapter/optimizer is called, including the synthetic convention fit.
    """
    CONFIG.validate()
    pairs = dataset.validate_dataset(manifest, verbose=False)
    if not 1 <= num_pairs <= len(pairs):
        raise ValueError(f"num_pairs must be between 1 and {len(pairs)}.")
    if any(int(row["num_points"]) != 1000 for row in pairs):
        raise ValueError("The fixed 1000-data-point Go-ICP comparison requires 1000 points per cloud.")
    versions = {name: metadata.version(name) for name in
                ("numpy", "scipy", "pymanopt", "autograd", "matplotlib", "open3d", "pycpd", "py-goicp")}
    if versions["pycpd"] != "2.0.0":
        raise ValueError("This benchmark requires validated pycpd 2.0.0.")
    for name in ("open3d", "pycpd"):
        registration._require_dependency(name)  # Imports outside fitting timers.
    registration._require_goicp()
    check = registration.load_goicp_convention_record(convention_record, CONFIG)
    signature = {
        "manifest": str(manifest.resolve()), "manifest_sha256": dataset.sha256(manifest),
        "dataset_identity": dataset.dataset_identity(pairs[0]),
        "method_settings": method_settings(), "package_versions": versions,
        "registration_source_sha256": dataset.sha256(ROOT / "emdb_baseline_comparison.py"),
        "driver_source_sha256": dataset.sha256(Path(__file__)),
        "generator_source_sha256": dataset.sha256(ROOT / "generate_emdb_resampling_pairs.py"),
        "diagnostics_source_sha256": dataset.sha256(ROOT / "emdb_resampling_diagnostics.py"),
        "thread_environment": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    return pairs, signature, check


def atomic_write(path: Path, write) -> None:
    """Flush/fsync a temporary sibling, then replace; keep the last good file.

    An interrupted native method cannot truncate a previous checkpoint. A hard
    process kill may leave an unused .tmp sibling; it is never read on resume.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_checkpoint(path: Path, rows: list[dict]) -> None:
    def write(stream):
        if rows:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    atomic_write(path, write)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, lambda stream: json.dump(value, stream, indent=2))


def load_checkpoint(resume_from: Path, pairs: list[dict], manifest: Path,
                    num_pairs: int, signature: dict) -> list[dict]:
    previous = json.loads((resume_from.parent / "baseline_run_metadata.json").read_text(encoding="utf-8"))
    saved = previous["reproducibility_signature"]
    legacy = saved.get("driver_source_sha256") == LEGACY_DRIVER_SHA256
    if legacy:
        # Only the known validated 9515 driver is migratable. Its algorithms,
        # packages, data hashes, settings and thread environment must still match.
        if (pairs[0]["emdb_id"] != "EMD-9515" or set(saved) != {
                "manifest", "manifest_sha256", "method_settings", "package_versions",
                "registration_source_sha256", "driver_source_sha256", "thread_environment"}
                or any(saved[key] != signature[key] for key in saved if key != "driver_source_sha256")):
            raise ValueError("Legacy checkpoint has incompatible data/settings/code/environment.")
    elif saved != signature:
        raise ValueError("Cannot reuse results with different frozen data, code, method settings, versions or thread environment.")
    rows = dataset.read_manifest(resume_from)
    validate_saved_rows(rows, pairs, manifest, num_pairs, allow_legacy=legacy)
    if legacy:
        # Enrich the new destination copy only; never alter the legacy CSV.
        for row in rows:
            p = next(p for p in pairs if p["pair_id"] == row["pair_id"])
            Xc, Yc, _, _ = prepare_pair(p, manifest)
            row.update(emdb_id=p["emdb_id"], map_file=str(Path(p["map_file"]).resolve()),
                       map_sha256=p["map_sha256"], contour_threshold=dataset.contour_threshold(p),
                       **geometry.pair_diagnostics(Xc, Yc))
            row.update({field: row.get(field, None) for field in EXTRA_FIELDS})
            if row["method"] == "goicp":
                t = np.array(json.loads(row["goicp_translation"]))
                row.update(goicp_translation_x=float(t[0]), goicp_translation_y=float(t[1]),
                           goicp_translation_z=float(t[2]),
                           goicp_translation_near_boundary_axes=json.dumps((np.minimum(t+.05, .05-t) <= 1e-4).tolist()))
    return rows


def aggregate_results(rows: list[dict]) -> list[dict]:
    """Preserve common statistics; obtain dataset identity from the trial rows."""
    identity = {key: rows[0][key] for key in ("emdb_id", "map_file", "map_sha256", "contour_threshold")}
    if any(any(row[key] != value for key, value in identity.items()) for row in rows):
        raise ValueError("Cannot aggregate different datasets/configurations.")
    return [dict(summary, **identity) for summary in registration.aggregate_results(rows)]


def run_experiment(manifest: Path, num_pairs: int, output_dir: Path,
                   convention_record: Path = DEFAULT_CONVENTION, resume_from: Path | None = None) -> list[dict]:
    trial_path = output_dir / "baseline_trial_results.csv"
    in_place = resume_from is not None and resume_from.resolve() == trial_path.resolve()
    if output_dir.exists() and any(p.is_file() for p in output_dir.iterdir()) and not in_place:
        raise ValueError("Output contains files; use a new destination or resume its exact trial CSV.")
    pairs, signature, check = preflight(manifest, num_pairs, convention_record)
    if in_place:
        old_metadata = json.loads((output_dir / "baseline_run_metadata.json").read_text(encoding="utf-8"))
        if old_metadata["reproducibility_signature"].get("driver_source_sha256") == LEGACY_DRIVER_SHA256:
            raise ValueError("Preserve legacy results: migrate only to a NEW destination.")
    rows = load_checkpoint(resume_from, pairs, manifest, num_pairs, signature) if resume_from else []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    run_metadata = {
        "protocol": "independent frozen finite resampling pairs; one imposed population rotation per pair",
        "metric_interpretation": "recovery of artificially imposed population rotation; no exact pointwise transformation between independent clouds",
        "num_pairs": num_pairs, "master_seed": int(pairs[0]["master_seed"]),
        "dataset_identity": dataset.dataset_identity(pairs[0]),
        "reproducibility_signature": signature, "goicp_convention_record": str(convention_record.resolve()),
        "goicp_convention_check_reused": check, "resume_from": str(resume_from.resolve()) if resume_from else None,
        "reused_rows": len(rows), "success_thresholds": "descriptive only: angle <1, <5, <10 degrees",
        "runtime_scope": "unchanged method adapters: setup/fitting; excludes imports, shared preprocessing, evaluation and plotting",
        "checkpoint_policy": "atomic CSV replace after each completed method; resume validates an ordered prefix",
    }
    metadata_path = output_dir / "baseline_run_metadata.json"
    write_json(metadata_path, run_metadata)
    write_checkpoint(trial_path, rows)
    try:
        for pair in pairs[:num_pairs]:
            completed = [r["method"] for r in rows if r["pair_id"] == pair["pair_id"]]
            for row in iter_pair_results(pair, manifest, completed_methods=completed):
                write_checkpoint(trial_path, rows + [row])
                rows.append(row)
                print(f"{row['pair_id']} {row['method']}: angle={row['rotation_angle_error_degrees']:.6f} deg, "
                      f"rotation={row['rotation_error']:.8f}, runtime={row['runtime_seconds']:.3f}s", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        expected = [(p["pair_id"], m) for p in pairs[:num_pairs] for m in registration.METHODS]
        failure = {"dataset_identity": dataset.dataset_identity(pairs[0]), "completed_rows": len(rows),
                   "next_uncheckpointed_pair_method": expected[len(rows)] if len(rows) < len(expected) else None,
                   "exception_type": type(exc).__name__, "message": str(exc),
                   "checkpoint": str(trial_path.resolve()), "reproducibility_signature": signature,
                   "elapsed_seconds": perf_counter() - started}
        if isinstance(exc, registration.GoICPRotationError):
            failure["goicp_rotation_diagnostics"] = exc.diagnostics
            write_json(output_dir / "goicp_rotation_failure.json", exc.diagnostics)
        write_json(output_dir / "benchmark_failure.json", failure)
        raise
    # Validate serialization against the frozen manifest before aggregation.
    rows = dataset.read_manifest(trial_path)
    validate_saved_rows(rows, pairs, manifest, num_pairs)
    if len(rows) != num_pairs * 4:
        raise RuntimeError("Incomplete paired benchmark.")
    numeric = [dict(r, **{key: float(r[key]) for key in registration.COMMON_METRICS}) for r in rows]
    write_checkpoint(output_dir / "baseline_aggregated_results.csv", aggregate_results(numeric))
    registration._write_csv(output_dir / "pairwise_results.csv", pairwise_table(numeric))
    save_plots(numeric, output_dir)
    status = smoke_status(rows)
    status["smoke_passed"] = status["cpd_all_converged"] and not status["goicp_any_translation_near_boundary"]
    (output_dir / "smoke_validation.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    # Recheck the frozen inputs after fitting, independent of adapter buffers.
    dataset.validate_dataset(manifest, verbose=False)
    run_metadata["invocation_elapsed_seconds"] = perf_counter() - started
    write_json(metadata_path, run_metadata)
    print(json.dumps(status, indent=2), flush=True)
    print(f"Saved {len(rows)} rows and four plots in {output_dir.resolve()}; elapsed={perf_counter()-started:.2f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--goicp-convention-record", type=Path, default=DEFAULT_CONVENTION)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        pairs, signature, _ = preflight(args.manifest, args.num_pairs, args.goicp_convention_record)
        if args.resume_from:
            cached = load_checkpoint(args.resume_from, pairs, args.manifest, args.num_pairs, signature)
        else:
            cached = []
        print(json.dumps({"preflight_passed": True, "no_registrations_run": True,
                          "validated_pairs": len(pairs), "requested_pairs": args.num_pairs,
                          "reusable_rows": len(cached), "reproducibility_signature": signature}, indent=2), flush=True)
    else:
        run_experiment(args.manifest, args.num_pairs, args.output_dir, args.goicp_convention_record, args.resume_from)


if __name__ == "__main__":
    main()
