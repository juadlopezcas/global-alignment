"""Read-only geometry diagnostics; no registration, RNG draws or tuning."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import mrcfile
import numpy as np

import emdb_baseline_comparison as registration
import generate_emdb_resampling_pairs as dataset


def covariance_diagnostics(points: np.ndarray) -> dict:
    """Sample covariance eigenvalues in descending order, adjacent absolute gaps."""
    centered = points - points.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / (len(points) - 1))[::-1]
    return {"covariance_eigenvalues": eigenvalues.tolist(),
            "pca_eigenvalue_gaps": (eigenvalues[:-1] - eigenvalues[1:]).tolist()}


def pair_diagnostics(Xc: np.ndarray, Yc: np.ndarray) -> dict:
    result = {}
    for name, points in (("Xc", Xc), ("Yc", Yc)):
        for key, value in covariance_diagnostics(points).items():
            result[f"{name}_{key}"] = json.dumps(value)
    return result


def describe_dataset(manifest: Path, rows: list[dict]) -> tuple[dict, list[dict]]:
    identity = dataset.dataset_identity(rows[0])
    population, _ = dataset.load_population(Path(identity["map_file"]), identity["contour_threshold"], identity["num_points"])
    with mrcfile.open(identity["map_file"]) as mrc:
        shape = list(mrc.data.shape)
        voxel_size = [float(mrc.voxel_size[key]) for key in ("x", "y", "z")]
        labels = [label.decode("ascii", errors="replace").strip() for label in mrc.header.label[:int(mrc.header.nlabl)]]
    summary = {**identity, "map_shape_zyx": shape, "voxel_size_xyz": voxel_size,
               "map_header_labels": labels, "num_pairs": len(rows), "num_eligible_voxels": len(population),
               "coordinate_bbox_min_xyz": population.min(axis=0).tolist(),
               "coordinate_bbox_max_xyz": population.max(axis=0).tolist(),
               "population_covariance": covariance_diagnostics(population),
               "covariance_convention": "centered sample covariance / (n-1); descending eigenvalues; adjacent absolute gaps",
               "normalized_scale_convention": "unchanged shared scale from center_and_normalize; diagnostics only"}
    pairs = []
    for row in rows:
        X = registration.load_xyz(dataset.resolve_cloud(manifest, row["source_file"]))
        Z = registration.load_xyz(dataset.resolve_cloud(manifest, row["target_unrotated_file"]))
        Y = Z @ np.array(json.loads(row["R_true"])).T
        Xc, Yc, scale = registration.center_and_normalize(X, Y)
        pairs.append({"emdb_id": row["emdb_id"], "pair_id": row["pair_id"],
                      "normalization_scale": scale,
                      "source_target_overlap_count": int(row["source_target_overlap_count"]),
                      **pair_diagnostics(Xc, Yc)})
    scales = [row["normalization_scale"] for row in pairs]
    summary["normalization_scale_range"] = [min(scales), max(scales)]
    return summary, pairs


def write_diagnostics(manifest: Path) -> dict:
    """Explicit invocation only: never writes to an old dataset during validation."""
    rows = dataset.validate_dataset(manifest, verbose=False)
    summary, pairs = describe_dataset(manifest, rows)
    paths = [manifest.parent / name for name in
             ("map_diagnostics.json", "pair_geometry_diagnostics.csv", "sampled-geometry.png")]
    if any(path.exists() for path in paths):
        raise ValueError("Diagnostics already exist; refusing to overwrite.")
    paths[0].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with paths[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for field, label, color in (("source_file", "X", "#2369a1"), ("target_unrotated_file", "Z", "#db7621")):
        cloud = registration.load_xyz(dataset.resolve_cloud(manifest, rows[0][field]))
        for ax, (i, j) in zip(axes, ((0, 1), (0, 2), (1, 2))):
            ax.scatter(cloud[:, i], cloud[:, j], s=3, alpha=.35, label=label, color=color)
            ax.set(xlabel="xyz"[i], ylabel="xyz"[j], aspect="equal")
    axes[0].legend()
    fig.suptitle(f"{rows[0]['emdb_id']}: pair_000 frozen X and unrotated Z")
    fig.savefig(paths[2], dpi=160); plt.close(fig)
    return summary
