# Go-ICP baseline setup and convention

The benchmark uses the Cython wrapper from
[aalavandhaann/go-icp_cython](https://github.com/aalavandhaann/go-icp_cython),
commit `4568dd976fc5a63246835edbc748f35bc483f409`, distribution `py-goicp 0.0.4`.
It calls the upstream Go-ICP implementation, without modifying its registration
algorithm. The existing ours, Open3D ICP, rigid CPD, RNG, preprocessing, and
common evaluation implementations are preserved.

## Tested Windows build

The compiled module is installed in `data/.venv-emdb-baselines` and imports as
`py_goicp`. This environment uses Python 3.11.8, Cython 0.29.37, autowrap 0.23.0,
and Visual Studio 2022 Build Tools (MSVC 19.44.35217, tools 14.44.35207).

`goicp_windows_build.patch` contains only two portability fixes:

- `setup.py`: use MSVC `/std:c++14` on Windows and omit the GCC linker flag.
- `src/matrix.hpp`: use standard `<stdint.h>` with modern MSVC to avoid the
  conflicting legacy `int8_t` declaration.

The wrapper source is under the ignored `data/go-icp_cython` directory. The patch
is kept outside that directory so the build can be reproduced. The installed
wrapper already works; the following is a reproduction procedure for a fresh
checkout/environment, not a step performed by the benchmark:

```powershell
# Run from the global-alignment repository root. Requires a C++ build toolchain.
$goicpPython = (Resolve-Path 'data/.venv-emdb-baselines/Scripts/python.exe').Path
& $goicpPython -m pip install 'Cython==0.29.37' 'autowrap==0.23.0'
git clone https://github.com/aalavandhaann/go-icp_cython.git data/go-icp_cython
git -C data/go-icp_cython checkout 4568dd976fc5a63246835edbc748f35bc483f409
git -C data/go-icp_cython apply ../../goicp_windows_build.patch
Push-Location data/go-icp_cython
& $goicpPython -B setup.py build_ext --inplace
Pop-Location
& $goicpPython -m pip install --no-index --no-deps --no-build-isolation ./data/go-icp_cython
& $goicpPython -c "import py_goicp; from importlib.metadata import version; print(version('py-goicp')); print(py_goicp.__file__)"
```

The original compiler errors were resolved by the two compatibility changes;
the successful build still emitted upstream numeric-conversion warnings.
No compilation or installation happens automatically inside the benchmark.
If loading fails, the script names the required wrapper/toolchain and the
[original C++ executable](https://github.com/yangjiaolong/Go-ICP) alternative.
An executable adapter is not needed by this successful Python build and is not
implemented; the script never silently switches backends.

## Inputs, rotation, and evaluation

`run_trial` generates one `R_true` and calls the original shared preprocessing
once. All four adapters receive the same read-only `Xc` and `Yc` arrays.
Go-ICP receives **model = target = Yc**, **data = source = Xc**. No additional
centering, scaling, point matching, or random draws are performed in `run_goicp`.
Coordinates are checked against `[-1, 1]^3`. The upstream `POINT3D` storage is
float32; the required native conversion roundoff is reported as
`goicp_max_coordinate_roundoff`.

The wrapper returns a matrix acting on column points. Therefore:

```text
R_raw = optimalRotation()       # no transpose or inverse at extraction
R_est, diagnostics = clean_goicp_rotation(R_raw)
X_aligned = Xc @ R_est.T + translation
```

Common metrics evaluate only `R_est` against `R_true`; the translation is
recorded separately. `goicp_final_error` is the native distance-transform-based
SSE returned by `Register()`, not an angular error or MMD. Ground truth is never
an input to any fitting adapter.

### Guarded numerical cleanup

The fourth real trial (zero-based index 3, seed 42) completed native search but
returned a matrix whose maximum orthogonality residual was `1.05965924e-6`,
slightly above the unchanged shared `1e-6` check. Its Frobenius residual was
`1.19322015e-6`, determinant `0.9999992909318903`, and largest singular-value
deviation from one `5.90456172e-7`. The wrapper returned Python float lists
(`np.asarray` dtype `float64`); native rotation arithmetic includes floats and
`ICP3D<float>`. The diagnostic supports tiny accumulated numerical drift, not
a transpose/inverse correction. Native search code and parameters are unchanged.

`clean_goicp_rotation` applies the same policy to every Go-ICP result:

1. Record the raw matrix, inferred dtype, return type, orthogonality residuals,
   determinant, and singular values before strict validation.
2. Require a finite 3x3 matrix, positive determinant, and each of Frobenius/max
   orthogonality residual, `abs(det-1)`, and maximum `abs(singular_value-1)` at
   most **5e-6**. Reflections or larger deviations are rejected.
3. If the raw matrix already passes `assert_rotation`, preserve it exactly.
4. Otherwise use `U @ diag(1,1,sign(det(U@Vt))) @ Vt`, require a projection
   distance at most **5e-6**, and verify the result with the unchanged strict
   SO(3) check. Use that rotation in common evaluation.

The guard was chosen after reporting the actual raw residuals, without using
ground-truth error to select it. It never expands automatically. Cleanup is
part of the Go-ICP adapter timer and never changes the native SSE diagnostic.
CSV fields include `goicp_raw_rotation`, `goicp_raw_dtype`,
`goicp_raw_return_type`, `goicp_raw_source`,
`goicp_raw_orthogonality_error_fro`, `goicp_raw_orthogonality_error_max`,
`goicp_raw_determinant`, `goicp_raw_singular_values`,
`goicp_projection_applied`, `goicp_projection_distance`, and
`goicp_cleanup_tolerance`. Other methods leave these fields blank.
Rejected matrices produce `goicp_rotation_failure_trial_<index>.json` through
the CLI, and the exception retains the raw diagnostics for direct callers.

Trial 4's projection distance was `5.96610247e-7`. Its projected Frobenius
orthogonality residual is `1.12675331e-15` and determinant
`1.0000000000000009`. Before/after Frobenius registration errors are
`0.0294753475856789` / `0.0294753505306464`. The raw trace-formula angle is
`1.1951670299` degrees; the projected benchmark angle is `1.1941928708` degrees.
The raw trace-formula value is a diagnostic, not an exact SO(3) geodesic angle
because the raw matrix is not exactly orthogonal.

Full capture and projection analysis are saved under
`numerical_results/emdb_goicp_trial4_diagnosis`. The native failed trial was
reproduced once; subsequent validation used that saved output, not another
search. No full benchmark was started during diagnosis or repair.

### Preserve and resume saved trials

`--resume-from <trial CSV>` validates the source configuration and complete,
ordered four-method trial prefix, checks saved ground truth/preprocessing and
metrics, and copies it into a **new output directory**. It then runs only
missing trial indices. The source CSV and metadata are never edited. Legacy
Go-ICP rows get raw diagnostics backfilled from their stored pre-cleanup
`R_est`; their rotations and common metrics must remain unchanged under the
new policy, otherwise resume is refused for explicit review.

The interrupted `numerical_results/emdb_baseline_comparison` CSV contains 12
rows (indices 0-2, all methods). All three saved Go-ICP rotations pass the
new policy unchanged. The following manual command resumes at **index 3**;
it was not executed during this repair. Trial 3's other methods were not saved
by the interrupted all-method trial, so that trial is run as a complete group.
Use a different fresh output directory if the one below already exists.

```powershell
Set-Location 'C:\Users\22898\Desktop\project\global-alignment'
# Match the interrupted run's recorded unset thread environment.
Remove-Item Env:OPENBLAS_NUM_THREADS, Env:OMP_NUM_THREADS, Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
& '.\data\.venv-emdb-baselines\Scripts\python.exe' -B .\emdb_baseline_comparison.py `
  --num-trials 20 --seed 42 `
  --goicp-dt-size 300 --goicp-num-data-points 1000 `
  --goicp-translation-min -0.05 --goicp-translation-width 0.10 `
  --goicp-mse-threshold 0.001 --goicp-trim-fraction 0 `
  --goicp-convention-record .\numerical_results\emdb_goicp_feasibility\final_convention_status.json `
  --resume-from .\numerical_results\emdb_baseline_comparison\baseline_trial_results.csv `
  --output-dir .\numerical_results\emdb_baseline_comparison_resumed
```

The benchmark requires a successful Go-ICP convention check. By default,
`validate_goicp_convention` runs it; `--goicp-convention-record` can instead reuse
a saved successful synthetic-check status JSON with exactly matching Go-ICP
parameters. The record path is saved in run metadata. Mismatched parameters,
failed/unfinished records, or failing recorded errors are rejected.

The test uses an independent asymmetric synthetic cloud of 80 points and a known 8-degree
rotation, followed by the existing centering/shared normalization. It checks
SO(3), rotation error, angular error, transformed points, and small translation.
A failure stops execution before any EMDB trial.

The repeated standalone DT=300 test passed on 2026-09-28: Frobenius rotation
error `2.7877113607832833e-9`, angular error `0` degrees at floating-point
precision, and translation norm `3.7987290021163126e-5`. Total runtime was
`12.2364 s`, including `12.2248 s` for DT construction. No transpose/inverse was
needed. The interrupted run's recovered result also passed (total `9.7030 s`).

## Parameters and timing

The initial, interrupted smoke run used these settings from the upstream example:

| Parameter | Value |
| --- | --- |
| MSE threshold | 0.001 |
| DT nodes per axis | 300 |
| DT expansion factor | 2.0 |
| Trim fraction / trimming enabled | 0 / false |
| EMDB source/data points | all 1000 |
| EMDB target/model points | all 1000 |
| Rotation angle-axis cube | minimum -3.1416 on each axis, width 6.2832 |
| Original translation cube | minimum -0.5 on each axis, width 1.0 |

The subsequent runtime-feasibility experiment changes the translation cube to
minimum **-0.05**, width **0.10** on each axis. These are now the adapter/CLI
defaults, exposed as `--goicp-translation-min` and `--goicp-translation-width`.
Both values are included in the trial diagnostics and run configuration.

These are fixed independently of ground truth. `--goicp-num-data-points 0`
means all points; an explicitly smaller positive count uses the first N source
rows in their existing order, and an oversized count uses all available rows.
The 80-point synthetic convention check thus uses all 80 points even with
the configured count 1000. DT size is never automatically reduced.

`runtime_seconds` and `goicp_runtime_seconds` include coordinate conversion,
native setup, DT construction, registration search, output extraction, and
object cleanup. `goicp_dt_runtime_seconds` and `goicp_search_runtime_seconds`
also report the two native phases separately. Imports, shared preprocessing,
evaluation, plotting, and the separate convention check are outside each
trial's method timer.

## Initial smoke-test status: stopped by request

On 2026-09-28 the standalone convention check passed, followed by the 1-trial
EMDB invocation below. The convention checks inside that invocation also
passed. The native real-data Go-ICP search did not finish before the user asked
to stop. Both smoke-test Python processes were stopped and their termination
was verified.

The last observed process elapsed time was **648.5 seconds (10 minutes 49
seconds)**, including startup, convention checks, and the other three methods.
This is a lower bound on the interrupted process duration, not a completed
Go-ICP per-trial runtime. The native log had reached registration search after
DT construction; no final rotation, translation, SSE, or method runtime was
returned. No parameter was reduced or changed.

`numerical_results/emdb_goicp_smoke_one/baseline_trial_results.csv` is empty:
the existing trial function returns its rows only after all methods finish.
Thus even the original three methods' results from this invocation were not
saved. The original three-method smoke results remain unchanged in their
separate directory. The interrupted directory preserves the run metadata and
`smoke_test_status.json`; it contains no aggregate CSV or comparison plots.

The **2-trial smoke test and 20-trial benchmark were not started**. Real-data
Go-ICP rotation/angle errors, SO(3) checks, translation norm, and complete runtime
remain unverified until the real-data search finishes. All **15** adapter,
convention, preservation, evaluation, and shared-protocol unit tests passed.
Before relying on a full benchmark, complete the requested 1-trial and 2-trial
checks. Resuming requires a fresh invocation; the native search has no saved
checkpoint.

That was the status at the initial interruption. The subsequent reduced-range
experiment below completed one real-data fit and validated its convention.

## Staged runtime feasibility on one EMDB trial

`emdb_goicp_feasibility.py` prepares trial 0, seed 42, once using the existing
trial RNG and preprocessing. Its arrays match `run_trial` exactly. It evaluates
only Go-ICP; Ours, ICP, CPD, and their original 1000-point inputs are unchanged.
The original convention is treated as already validated. There is no repeat of
the original wide-translation EMDB fit or its convention check.

The conditional sequence is fixed before fitting:

| Stage | DT size | Source/data points | Translation minimum | Width | MSE | Trim |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 300 | 1000 | -0.05 | 0.10 | 0.001 | 0 |
| 2 | 300 | 500 | -0.05 | 0.10 | 0.001 | 0 |
| 3 | 300 | 200 | -0.05 | 0.10 | 0.001 | 0 |
| 4 | 150 | 200 | -0.05 | 0.10 | 0.001 | 0 |
| 5 | 100 | 200 | -0.05 | 0.10 | 0.001 | 0 |

Every stage keeps all 1000 target/model points and the unchanged shared
normalization. Source subsets use the first N rows in their original file
order, with no new RNG, centering, or normalization. Rotation search bounds and
DT expansion remain unchanged.

Each native fit runs in a separate process with a **180-second hard timeout**,
including worker startup, DT construction and search. A timeout terminates and
joins that worker before the next stage starts. The controller selects the
first completed stage based only on completion within the fixed time budget,
before computing errors against `R_true`. It does not claim that untested later
stages would be slower or less accurate. The native fitting worker receives no
EMDB ground truth.

The selected configuration, if any, is then checked with the existing synthetic
convention test, using the same parameter values (the synthetic cloud has only
80 points, so its configured source-count cap naturally uses all 80). No
additional EMDB trial, 2-trial comparison, or 20-trial comparison is launched.

`feasibility_results.csv` records every tested stage, including timeouts with
blank error fields. Completed runtimes are the full adapter timer; timeout
runtimes are observed worker elapsed time through termination, not completed
fit runtimes. Separate JSON files preserve input hashes, settings, worker exit
status, native rotations/diagnostics, evaluation, convention validation, and
the completion-based recommendation. Existing output directories are rejected
to prevent overwriting earlier attempts.

```powershell
# With the Python path and thread limits set as below; choose a NEW directory.
& $goicpPython -B emdb_goicp_feasibility.py --timeout-seconds 180 --output-dir numerical_results/emdb_goicp_feasibility
```

The feasibility tests exercise actual worker termination, the fixed stage
sequence, equality with the original trial's arrays, and stopping at completion
even when the reported ground-truth rotation error is large:

```powershell
& $goicpPython -B -m unittest test_emdb_goicp_feasibility -v
```

## Running the benchmark later

Use separate output directories to preserve earlier results. Both smoke runs
and the full command use seed 42 and the original data/protocol defaults.
Each invocation performs the existing ICP/CPD checks and either runs the Go-ICP
check or reuses the explicitly supplied matching validation record.
The commands below now use the reduced translation-cube defaults; choose the
validated DT/source-count settings from the feasibility results before running
additional benchmark trials. These commands are not run by the feasibility
script.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$goicpPython = 'C:\Users\22898\Desktop\project\global-alignment\data\.venv-emdb-baselines\Scripts\python.exe'
& $goicpPython -B emdb_baseline_comparison.py --num-trials 2 --seed 42 --goicp-mse-threshold 0.001 --goicp-dt-size 300 --goicp-trim-fraction 0 --goicp-num-data-points 1000 --goicp-translation-min -0.05 --goicp-translation-width 0.10 --goicp-convention-record numerical_results/emdb_goicp_feasibility/final_convention_status.json --output-dir numerical_results/emdb_goicp_smoke_two
```

The following command is for the later full benchmark, after successful smoke
tests; it has **not** been run:

```powershell
& $goicpPython -B emdb_baseline_comparison.py --num-trials 20 --seed 42 --goicp-mse-threshold 0.001 --goicp-dt-size 300 --goicp-trim-fraction 0 --goicp-num-data-points 1000 --goicp-translation-min -0.05 --goicp-translation-width 0.10 --goicp-convention-record numerical_results/emdb_goicp_feasibility/final_convention_status.json --output-dir numerical_results/emdb_baseline_comparison_goicp
```

Each result directory contains trial and aggregate CSVs, run metadata with
versions/settings/convention results, and the three existing comparison plots
extended to four methods. Aggregate sample standard deviations for a single
trial are intentionally undefined (NaN), following the existing aggregation.

Tests for the common protocol and Go-ICP adapter can be run using:

```powershell
& $goicpPython -B -m unittest test_emdb_baseline_comparison -v
```
