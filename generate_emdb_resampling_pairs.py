"""Freeze independent EMDB resamples using map.py's exact sampling rule.

No origin/axis correction, density weighting, interpolation, noise or coordinate
perturbation. Uniform sampling without replacement WITHIN each cloud; overlap
between clouds/pairs is permitted. All stored XYZ coordinates use %.4f.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
from importlib import metadata
import json
from pathlib import Path

import mrcfile
import numpy as np

from emdb_baseline_comparison import assert_rotation, load_xyz, random_rotation

ROOT = Path(__file__).resolve().parent
DEFAULT_MAP = ROOT / "data/emd_9515.map"
DEFAULT_OUTPUT = ROOT / "data/emdb_9515_resampling_pairs"
THRESHOLD = 0.035
NUM_POINTS = 1000
NUM_PAIRS = 20
STREAM_NAMES = ("source_sampling_seed", "target_sampling_seed", "rotation_seed", "ours_fit_seed")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_seeds(master_seed: int, pair_index: int) -> dict[str, int]:
    """Explicit stored integer seeds: child 0=X, 1=Z, 2=rotation, 3=Ours fit.

    Separate SeedSequence([master_seed, pair_index]).spawn(4) for each pair;
    generate_state(1, uint64) gives each child its stored default_rng seed.
    No Python hash(), shared mutable RNG, or method-dependent draw consumption.
    """
    children = np.random.SeedSequence([master_seed, pair_index]).spawn(4)
    return {name: int(child.generate_state(1, dtype=np.uint64)[0])
            for name, child in zip(STREAM_NAMES, children)}


def load_population(map_file: Path, contour_threshold: float = THRESHOLD,
                    num_points: int = NUM_POINTS) -> tuple[np.ndarray, float]:
    """Authoritative map.py math, including scalar voxel_size.x and (x,y,z)."""
    if not np.isfinite(contour_threshold) or num_points < 1:
        raise ValueError("Threshold must be finite and num_points positive.")
    with mrcfile.open(map_file, mode="r") as mrc:
        density_data = mrc.data
        voxel_size = mrc.voxel_size.x
        z_idx, y_idx, x_idx = np.where(density_data > contour_threshold)
        point_cloud = np.column_stack((x_idx, y_idx, z_idx)) * voxel_size
    if len(point_cloud) < num_points:
        raise ValueError(f"Only {len(point_cloud)} eligible voxels at threshold {contour_threshold}; need {num_points}. No settings changed.")
    if not np.isfinite(point_cloud).all() or not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("Map coordinates/voxel_size.x are invalid.")
    return point_cloud, float(voxel_size)


def sample_cloud(population: np.ndarray, rng: np.random.Generator,
                 num_points: int = NUM_POINTS) -> np.ndarray:
    indices = rng.choice(len(population), num_points, replace=False)
    return population[indices]


def xyz_bytes(points: np.ndarray) -> bytes:
    stream = io.StringIO(newline="")
    np.savetxt(stream, points, fmt="%.4f")
    return stream.getvalue().encode("ascii")


def point_set(points: np.ndarray) -> set[tuple]:
    return set(map(tuple, points))


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def resolve_cloud(manifest: Path, filename: str) -> Path:
    """Portable relative manifest paths, contained in the frozen dataset folder."""
    path = (manifest.parent / filename).resolve()
    if path.parent != manifest.parent.resolve() or Path(filename).name != filename:
        raise ValueError("Manifest cloud paths must name files in its own directory.")
    return path


def contour_threshold(row: dict) -> float:
    """Accept the frozen legacy 'threshold' column without rewriting old files."""
    value = float(row.get("contour_threshold", row.get("threshold", "nan")))
    if not np.isfinite(value) or ("threshold" in row and float(row["threshold"]) != value):
        raise ValueError("Missing/invalid or inconsistent contour threshold.")
    return value


def dataset_identity(row: dict) -> dict:
    return {"emdb_id": row["emdb_id"], "map_file": str(Path(row["map_file"]).resolve()),
            "map_sha256": row["map_sha256"], "contour_threshold": contour_threshold(row),
            "num_points": int(row["num_points"]), "master_seed": int(row["master_seed"])}


def validate_dataset(manifest: Path, *, verbose: bool = True) -> list[dict]:
    """Validate frozen files; never redraw sampling or regenerate stored rotations."""
    rows = read_manifest(manifest)
    if not rows or [r["pair_id"] for r in rows] != [f"pair_{i:03d}" for i in range(len(rows))]:
        raise ValueError("Expected nonempty distinct, ordered pair IDs.")
    identity = dataset_identity(rows[0])
    if not identity["emdb_id"].startswith("EMD-") or identity["num_points"] < 1 or identity["master_seed"] < 0:
        raise ValueError("Invalid dataset ID, point count or master seed.")
    num_points = identity["num_points"]
    map_file = Path(rows[0]["map_file"])
    population, voxel_size = load_population(map_file, identity["contour_threshold"], num_points)
    map_hash = sha256(map_file)
    # Same decimal formatting as the disk files, including rounding.
    eligible = point_set(np.char.mod("%.4f", population).astype(float))
    all_hashes, all_sets, seeds_seen = set(), set(), set()
    if verbose:
        print(f"Eligible voxels={len(population)}; pairs={len(rows)}; voxel_size.x={voxel_size:.10g}", flush=True)
    for index, row in enumerate(rows):
        if (dataset_identity(row) != identity or int(row["num_eligible_voxels"]) != len(population)
                or float(row["voxel_size_used"]) != voxel_size or Path(row["map_file"]).resolve() != map_file.resolve()
                or row["map_sha256"] != map_hash):
            raise ValueError(f"Sampling protocol mismatch: {row['pair_id']}.")
        expected = stream_seeds(int(row["master_seed"]), index)
        for name in STREAM_NAMES:
            seed = int(row[name])
            if seed != expected[name] or seed in seeds_seen:
                raise ValueError(f"Seed mismatch/reuse: {row['pair_id']} {name}.")
            seeds_seen.add(seed)
        sets = []
        for field, hash_field in (("source_file", "source_sha256"), ("target_unrotated_file", "target_sha256")):
            path = resolve_cloud(manifest, row[field])
            digest = sha256(path)
            cloud = load_xyz(path)
            points = point_set(cloud)
            if cloud.shape != (num_points, 3) or not np.isfinite(cloud).all() or len(points) != num_points:
                raise ValueError(f"Invalid cloud shape/values/duplicates: {path}.")
            if not points.issubset(eligible):
                raise ValueError(f"Cloud contains coordinates outside the reference thresholded population: {path}.")
            unordered = frozenset(points)
            if digest != row[hash_field] or digest in all_hashes or unordered in all_sets:
                raise ValueError(f"Hash mismatch or accidentally identical cloud: {path}.")
            all_hashes.add(digest)
            all_sets.add(unordered)
            sets.append(points)
        overlap = len(sets[0] & sets[1])  # Overlap is recorded, never rejected.
        if overlap != int(row["source_target_overlap_count"]):
            raise ValueError("Stored overlap diagnostic does not match files.")
        R_true = np.array(json.loads(row["R_true"]), dtype=float)
        if R_true.shape != (3, 3):
            raise ValueError("Manifest rotation must be 3 x 3.")
        assert_rotation(R_true)
        if verbose:
            print(f"{row['pair_id']}: overlap={overlap}; seeds={json.dumps(expected)}; "
                  f"X={resolve_cloud(manifest, row['source_file'])}; Z={resolve_cloud(manifest, row['target_unrotated_file'])}", flush=True)
    return rows


def generate_pairs(map_file: Path = DEFAULT_MAP, output_dir: Path = DEFAULT_OUTPUT,
                   master_seed: int = 42, *, emdb_id: str = "EMD-9515",
                   contour_threshold: float = THRESHOLD, num_points: int = NUM_POINTS,
                   num_pairs: int = NUM_PAIRS, verbose: bool = True) -> Path:
    if master_seed < 0 or num_pairs < 1 or num_points < 1 or not emdb_id.startswith("EMD-"):
        raise ValueError("Seed must be nonnegative, counts positive and ID EMD-prefixed.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Frozen dataset output must be new/empty; no existing pairs will be overwritten.")
    population, voxel_size = load_population(map_file, contour_threshold, num_points)
    map_hash = sha256(map_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(num_pairs):
        pair_id = f"pair_{index:03d}"
        seeds = stream_seeds(master_seed, index)
        X = sample_cloud(population, np.random.default_rng(seeds["source_sampling_seed"]), num_points)
        Z = sample_cloud(population, np.random.default_rng(seeds["target_sampling_seed"]), num_points)
        R_true = random_rotation(3, np.random.default_rng(seeds["rotation_seed"]))
        x_file, z_file = output_dir / f"{pair_id}_X.xyz", output_dir / f"{pair_id}_Z.xyz"
        x_file.write_bytes(xyz_bytes(X))
        z_file.write_bytes(xyz_bytes(Z))
        X_disk, Z_disk = load_xyz(x_file), load_xyz(z_file)
        rows.append({
            "pair_id": pair_id, "emdb_id": emdb_id, "map_file": str(map_file.resolve()),
            "threshold": contour_threshold, "num_eligible_voxels": len(population), "num_points": num_points,
            "voxel_size_used": voxel_size, "source_file": x_file.name, "target_unrotated_file": z_file.name,
            "master_seed": master_seed, **seeds, "R_true": json.dumps(R_true.tolist()),
            "source_sha256": sha256(x_file), "target_sha256": sha256(z_file), "map_sha256": map_hash,
            "source_target_overlap_count": len(point_set(X_disk) & point_set(Z_disk)),
        })
        # Preserve byte-for-byte regeneration of the legacy default manifest.
        # New configurations also expose the explicitly named threshold column.
        if not (emdb_id == "EMD-9515" and map_file.resolve() == DEFAULT_MAP.resolve()
                and contour_threshold == THRESHOLD and num_points == NUM_POINTS and num_pairs == NUM_PAIRS):
            rows[-1]["contour_threshold"] = contour_threshold
    manifest = output_dir / "resampling_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = {
        "seed_derivation": "SeedSequence([master_seed, pair_index]).spawn(4); child.generate_state(1, uint64)[0] -> stored integer seed -> default_rng(seed)",
        "stream_order": list(STREAM_NAMES), "sampling": f"rng.choice(num_eligible_voxels, {num_points}, replace=False); no density weights",
        "coordinate_rule": f"z,y,x=np.where(mrc.data>{contour_threshold}); column_stack((x,y,z))*mrc.voxel_size.x; no origin/axis correction",
        "xyz_format": "%.4f", "relative_cloud_paths": True, "rotation_function": "emdb_baseline_comparison.random_rotation",
        "rotation_source_sha256": sha256(ROOT / "emdb_baseline_comparison.py"),
        "reference_map_py_sha256": sha256(ROOT / "map.py"),
        "package_versions": {name: metadata.version(name) for name in ("numpy", "mrcfile")},
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    validate_dataset(manifest, verbose=verbose)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-file", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--emdb-id", default="EMD-9515")
    parser.add_argument("--contour-threshold", type=float, default=THRESHOLD)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--num-pairs", type=int, default=NUM_PAIRS)
    parser.add_argument("--seed", "--master-seed", dest="master_seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_dataset(args.output_dir / "resampling_manifest.csv")
    else:
        manifest = generate_pairs(args.map_file, args.output_dir, args.master_seed,
                                  emdb_id=args.emdb_id, contour_threshold=args.contour_threshold,
                                  num_points=args.num_points, num_pairs=args.num_pairs)
        print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
