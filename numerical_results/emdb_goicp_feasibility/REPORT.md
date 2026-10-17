# One-trial Go-ICP runtime feasibility

Stage 1 completed within the fixed 180-second deadline. The conditional
sequence therefore stopped without testing source-count or DT reductions.
Use **DT=300, all 1000 source points, translation minimum=-0.05 and width=0.10,
MSE=0.001, trim fraction=0** as the validated practical configuration from
this run. It is the first feasible configuration in the requested sequence;
untested smaller configurations might be faster, so this does not establish
the globally cheapest configuration.

| dt_size | num_data_points | translation_width | mse_threshold | completed_or_timeout | runtime_seconds | rotation_angle_error_degrees | rotation_error | translation_norm | final_goicp_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 1000 | 0.10 | 0.001 | completed | 159.136911 | 1.579288 | 0.038980212 | 0.027243119 | 3.127115011 |

Full-precision values are in `feasibility_results.csv`. Native final Go-ICP
error is distance-transform SSE. Runtime includes native adapter setup, DT
construction, search, extraction, and cleanup: DT took **15.706571 s** and
search took **143.394083 s**. Total worker elapsed time, also including Python
startup, was **159.847432 s**. The worker exited normally with code 0.

The estimated rotation passed the existing SO(3) checks without projection:
determinant **1.0000000150**, Frobenius orthogonality error **4.15e-8**.
Translation was `[-0.018750000745, -0.006250000093, -0.018750000745]` in the
shared-normalized coordinate system. `optimalRotation()` was used directly;
row alignment remains `Xc @ R_est.T + translation`.

## Selection and convention validation

The controller selected the configuration based on completion, **before**
computing the reported errors against ground truth. No error-dependent tuning
was performed. Trial 0 (seed 42) was prepared once with exactly the benchmark's
RNG and centering/shared normalization; all 1000 source and 1000 target points
were used. The native fitting worker received no EMDB ground truth.

After selection, the synthetic convention test was rerun with these exact
configured parameters and passed: rotation error **2.787711e-9**, angular error
**0 degrees** at numerical precision, translation norm **3.798729e-5**, runtime
**16.453032 s**. As in the existing synthetic test, it has 80 points, so the
configured 1000-point cap uses all 80. Its SO(3), transformed-point, and
small-translation checks all passed.

There was exactly one completed real EMDB configuration, followed only by the
synthetic check. The previous wide-translation run was not repeated. Neither
the 2-trial nor 20-trial comparison was run. One trial establishes feasibility
for this trial; it does not guarantee similar runtimes on other rotations.

## Code and preservation checks

The baseline adapter now exposes `--goicp-translation-min` and
`--goicp-translation-width`, defaults -0.05/0.10, and records both bounds.
`emdb_goicp_feasibility.py` implements the fixed staged sequence and hard worker
timeouts. It refuses to reuse existing result directories. Original benchmark
methods, RNG, preprocessing, and common evaluation functions are unchanged.

All 20 relevant tests passed: 16 baseline/adapter tests and 4 feasibility tests.
These include actual process termination on timeout, identical trial inputs,
and stopping at the first completion regardless of reported ground-truth error.
All 89 protected pre-existing data, source, wrapper patch, and result files
matched their pre-experiment hashes.

Additional JSON artifacts preserve configuration/input hashes, native output,
SO(3) diagnostics, worker status, convention validation, and the recommendation.
