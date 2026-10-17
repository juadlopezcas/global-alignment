# Frozen multi-map EMDB independent-resampling preparation

Prepared 20 pairs per map, 1000 points per cloud, master seed 42. No real
registration, synthetic convention re-run, smoke benchmark or full benchmark
was performed during this extension. All four registration adapters and their
parameters remain in the unchanged `emdb_baseline_comparison.py`.

## Maps and diagnostics

All maps were already available locally with matching EMDB header labels.
`data/EMD-3342.map` and `data/EMD-3388.map` were copied byte-for-byte to the
requested underscore filenames; originals were preserved. The existing
`data/emd_3802.map` was used directly. No substitute maps or downloads were used.

| EMDB | Threshold | Eligible voxels | Shape (z,y,x) | Voxel size (x=y=z) | Shared normalization scale range |
|---|---:|---:|---|---:|---|
| EMD-3342 | 0.015 | 107192 | 256 x 256 x 256 | 1.3150000572 | 79.841549–85.014143 |
| EMD-3802 | 0.030 | 70868 | 256 x 256 x 256 | 1.3200000525 | 103.794539–107.744871 |
| EMD-3388 | 0.080 | 193817 | 284 x 284 x 284 | 1.3600000143 | 126.062007–132.430105 |

Coordinate bounding boxes and population covariance diagnostics:

| EMDB | Minimum xyz | Maximum xyz | Descending covariance eigenvalues | Adjacent eigenvalue gaps |
|---|---|---|---|---|
| EMD-3342 | (118.350005, 111.775005, 109.145005) | (209.085009, 241.960011, 235.385010) | (883.051075, 470.590117, 301.059122) | (412.460957, 169.530996) |
| EMD-3802 | (96.360004, 97.680004, 117.480005) | (248.160010, 244.200010, 231.000009) | (1839.541964, 673.813810, 312.428253) | (1165.728153, 361.385557) |
| EMD-3388 | (84.320001, 112.880001, 85.680001) | (308.720003, 285.600003, 285.600003) | (3073.091097, 1772.817617, 782.835361) | (1300.273480, 989.982255) |

Covariance uses centered sample covariance divided by n-1. These diagnostic
eigenvalues/gaps do not replace the weighted PCA objective. Normalization scale
is the maximum centered radius across both clouds, as returned unchanged by
`emdb_baseline_comparison.center_and_normalize`; dividing by it gives joint
maximum radius 1. Each dataset's `pair_geometry_diagnostics.csv` records this
scale and the normalized Xc/Yc covariance eigenvalues and gaps for all 20 pairs.
`map_diagnostics.json` records full-precision population diagnostics, header
labels, voxel sizes and hashes. The three `sampled-geometry.png` files show
pair_000 X/Z coordinate projections and were visually checked.

## Protocol and compatibility

`generate_emdb_resampling_pairs.load_population` retains exactly
`z,y,x=np.where(mrc.data>threshold)` followed by
`column_stack((x,y,z))*mrc.voxel_size.x`. `sample_cloud` retains uniform
`rng.choice(len(population), num_points, replace=False)`. No origin correction,
axis remapping, density weights, interpolation, perturbation or forced
disjointness was added. XYZ serialization remains `%.4f`.

`stream_seeds` is unchanged: `SeedSequence([42,pair_index]).spawn(4)`, then
stored uint64 integer seeds and separate default_rng streams. Corresponding
pair indices therefore intentionally reuse the same four seeds and R_true
across maps; only the map and threshold affect sampling. Source/target streams
remain independent and overlap is allowed/recorded. All 60 pairs have finite
1000x3 clouds with unique coordinates, no accidentally identical clouds, valid
SO(3) ground truth, verified hashes and threshold-population membership.

Default EMD-9515 regeneration retains its legacy manifest schema (`threshold`)
and was verified byte-identical to the original frozen files and manifest.
New configurations include `contour_threshold` as well. Validation reads either
schema and rejects conflicting threshold aliases. The generator accepts
`--seed` and its existing `--master-seed` alias. A nonempty output dataset is
never overwritten. The generator supports configurable point/pair counts; the
controlled four-method benchmark explicitly requires 1000 points to preserve
its fixed Go-ICP setting.

`emdb_independent_resampling_comparison.prepare_pair`, its four adapter call
arguments, and `method_settings` were checked against the pre-change source.
Preprocessing still loads frozen X/Z/R_true, forms Y=Z@R_true.T, centers and
jointly normalizes once, and passes the same read-only arrays to all methods.
No ground truth enters fitting. Only recovery of the imposed rotation is
evaluated; native objectives are diagnostic fields, not comparative scores.

Settings remain: Ours sigma=1, weighted SO(3) PCA (200 iterations), best greedy
search (100 accepted moves, tolerance 1e-12), one TrustRegions refinement
(100 iterations, min gradient norm 1e-6); ICP identity initialization and
correspondence distance 2 with existing Open3D defaults; pycpd 2.0.0,
max_iterations=1000, tolerance=.001, w=0, existing scale/identity behavior and
no history callback; Go-ICP DT=300, 1000 data points, MSE=.001, trim=0,
translation minimum=-.05, width=.10, DT expansion=2, SO(3) guard=5e-6.
The successful saved Go-ICP convention record is reused, not rerun.

## Checkpoints and validation

`atomic_write`/`write_checkpoint` write a temporary sibling, flush and fsync,
then replace the trial CSV after each completed method. `load_checkpoint` and
`validate_saved_rows` check dataset/map/manifest identity, file and array hashes,
seeds, stored ground truth, rotation metrics, settings, code hashes, package
versions and thread environment before reusing an ordered pair/method prefix.
Missing methods only are fitted; duplicates, reordered rows and mixed
configurations are rejected before any checkpoint is overwritten. Metadata
must remain next to the CSV. Resume supports the same output folder or a fresh
destination. Known legacy EMD-9515 checkpoints can migrate only to a fresh
destination; completed old files are never rewritten.

A fitting exception or KeyboardInterrupt records `benchmark_failure.json`
with the next uncheckpointed pair/method and actual error, retaining earlier
rows. Go-ICP rotation failures additionally retain their raw diagnostics.
No fabricated metric row is written and no native timeout is introduced. A
hard process kill cannot write a failure log, but retains the last atomic
checkpoint; an abandoned temporary sibling is ignored on resume.

The 15 tests passed via:

```powershell
& .\data\.venv-emdb-baselines\Scripts\python.exe -B -m unittest test_emdb_independent_resampling test_emdb_multi_map -v
```

Tests regenerated all 60 new pairs and compared every XYZ file and manifest
byte-for-byte. All original tests passed, including EMD-9515 default byte
reproduction. All 80 completed EMD-9515 benchmark rows were validated without
refitting. SHA256 checks confirmed all 90 protected old files unchanged.
New tests prohibit real fits, exercise mocked native failure, atomic replace
failure, in-place resume, skipping completed methods, duplicate/wrong-map/
wrong-setting rejection, and all four final plots. CLI preflight passed for
each new 20-pair manifest. Each dataset has a `benchmark_preflight.json` record.

## Files

Modified: `generate_emdb_resampling_pairs.py`,
`emdb_independent_resampling_comparison.py`,
`test_emdb_independent_resampling.py` (only the mock ICP diagnostic).

Added source/documentation: `emdb_resampling_diagnostics.py`,
`test_emdb_multi_map.py`, this document.

Added maps: `data/emd_3342.map`, `data/emd_3388.map` (byte-identical copies).

Each of `data/emdb_3342_resampling_pairs`,
`data/emdb_3802_resampling_pairs`, `data/emdb_3388_resampling_pairs` contains:

- `pair_000_X.xyz` through `pair_019_X.xyz` (20 files);
- `pair_000_Z.xyz` through `pair_019_Z.xyz` (20 files);
- `resampling_manifest.csv`, `dataset_metadata.json`;
- `map_diagnostics.json`, `pair_geometry_diagnostics.csv`;
- `sampled-geometry.png`, `benchmark_preflight.json`.

That is 120 XYZ files and 18 metadata/diagnostic files. Audit artifacts:
`data/multi_emdb_preservation.json`, `data/multi_emdb_map_inventory.json`,
`data/independent_comparison_before_multi.py`, `data/generator_before_multi.py`,
and `data/multi_emdb_generated_files.json` (explicit generated file inventory).
Temporary test datasets were removed by the tests. No new trial/aggregate
benchmark CSVs or benchmark comparison plots have been generated.

## Manual full benchmarks and resume (PowerShell)

Run this setup once. Thread environment matches the validated preflight;
keep it the same when resuming.

```powershell
Set-Location 'C:\Users\22898\Desktop\project\global-alignment'
$benchmarkPython = 'C:\Users\22898\Desktop\project\global-alignment\data\.venv-emdb-baselines\Scripts\python.exe'
$benchmarkScript = 'C:\Users\22898\Desktop\project\global-alignment\emdb_independent_resampling_comparison.py'
$env:OPENBLAS_NUM_THREADS = $null
$env:OMP_NUM_THREADS = $null
$env:MKL_NUM_THREADS = $null
```

Full EMD-3342:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3342_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3342_independent_resampling
```

Full EMD-3802:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3802_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3802_independent_resampling
```

Full EMD-3388:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3388_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3388_independent_resampling
```

Resume EMD-3342:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3342_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3342_independent_resampling --resume-from numerical_results/emdb_3342_independent_resampling/baseline_trial_results.csv
```

Resume EMD-3802:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3802_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3802_independent_resampling --resume-from numerical_results/emdb_3802_independent_resampling/baseline_trial_results.csv
```

Resume EMD-3388:

```powershell
& $benchmarkPython -B $benchmarkScript --manifest data/emdb_3388_resampling_pairs/resampling_manifest.csv --num-pairs 20 --output-dir numerical_results/emdb_3388_independent_resampling --resume-from numerical_results/emdb_3388_independent_resampling/baseline_trial_results.csv
```

Append `--preflight-only` to any of these commands to validate without fitting
or writing to the benchmark output folder. The full commands produce the
normal long-form trial CSV, common-metric aggregate CSV, pairwise CSV, metadata,
validation JSON and four established angle/runtime comparison plots. New trial
and aggregate rows include dataset/map identity and contour threshold; trials
also include normalized geometry diagnostics, CPD iterations/convergence, and
Go-ICP translation components and boundary flags. No parameters adapt to map
difficulty or evaluation errors.
