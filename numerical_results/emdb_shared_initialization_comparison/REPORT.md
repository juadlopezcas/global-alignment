# Controlled initialization comparison: EMD-9515

20 consecutive frozen independent-resampling pairs; n=1000.
No map samples or imposed rotations were regenerated. X/Z quantization and
Y=Z@R_true.T, separate centering and shared maximum-radius normalization are
the original functions. All new refinements share the same read-only Xc/Yc
values per pair; CPD pre-rotation only represents its supplied initial rotation.
Ground truth is used only in result evaluation, never passed to a fitter.

## Accuracy and computational cost

| Condition | Mean +/- SD angle (deg) | Median (deg) | <1 / <5 / <10 deg | Mean refinement or global runtime (s) | Mean total pipeline (s) |
|---|---:|---:|---|---:|---:|
| mmd_identity | 116.975345 +/- 83.659119 | 176.305496 | 0.00 / 0.15 / 0.30 | 3.1041 | 3.1041 |
| icp_identity | 126.244231 +/- 47.403004 | 132.649798 | 0.00 / 0.05 / 0.05 | 0.0071 | 0.0071 |
| cpd_identity | 118.441470 +/- 71.746578 | 139.448884 | 0.15 / 0.25 / 0.25 | 12.9581 | 12.9581 |
| mmd_shared | 6.369077 +/- 2.917067 | 5.757695 | 0.00 / 0.45 / 0.90 | 0.9593 | 1.2833 |
| icp_shared | 1.267184 +/- 0.621817 | 1.113221 | 0.45 / 1.00 / 1.00 | 0.0045 | 0.3285 |
| cpd_shared | 1.440203 +/- 0.633246 | 1.270006 | 0.35 / 1.00 / 1.00 | 7.9123 | 8.2363 |
| goicp_global | 1.389571 +/- 0.664699 | 1.489530 | 0.30 / 1.00 / 1.00 | 129.5805 | 129.5805 |

Mean common-initialization angle: 6.621851943 degrees. Identity-to-shared
mean angle reductions: MMD 110.606267060,
ICP 124.977046305,
CPD 117.001266151 degrees.
These within-method contrasts quantify sensitivity to initialization on these
pairs; they do not decompose all between-method differences into causal shares.

| First - second | Mean paired angle difference (deg) | First lower / second lower / ties |
|---|---:|---|
| mmd_shared - mmd_identity | -110.606267 | 18 / 2 / 0 |
| icp_shared - icp_identity | -124.977046 | 20 / 0 / 0 |
| cpd_shared - cpd_identity | -117.001266 | 18 / 2 / 0 |
| icp_shared - mmd_shared | -5.101893 | 20 / 0 / 0 |
| cpd_shared - mmd_shared | -4.928874 | 20 / 0 / 0 |
| icp_shared - cpd_shared | -0.173019 | 11 / 9 / 0 |
| mmd_shared - shared_initial | -0.252774 | 10 / 10 / 0 |
| icp_shared - shared_initial | -5.354668 | 20 / 0 / 0 |
| cpd_shared - shared_initial | -5.181648 | 20 / 0 / 0 |
| mmd_shared - goicp_global | 4.979507 | 0 / 20 / 0 |
| icp_shared - goicp_global | -0.122386 | 12 / 8 / 0 |
| cpd_shared - goicp_global | 0.050633 | 7 / 13 / 0 |

Negative paired first-minus-second difference favors the first method. Sample
SDs, medians and per-pair differences are provided; no significance claims or
general-superiority claims follow from these 20 pairs. Raw rotation error is
||R_true-R_est||_F. Angular errors use the unchanged clipped trace/arccos metric
without signed-permutation equivalence.

MMD reduces empirical rotation-only MMD but worsens angular error in
10 shared-start pairs: ['pair_000', 'pair_001', 'pair_004', 'pair_008', 'pair_010', 'pair_012', 'pair_013', 'pair_015', 'pair_018', 'pair_019']. Initial and final MMD are
recorded for MMD only; native ICP/CPD/Go-ICP objectives are not comparative
accuracy metrics. A lower empirical loss need not improve imposed-rotation
recovery on independent finite clouds. CPD converged by its native tolerance
in 40/40 identity/shared rows.
Open3D 0.19's result does not expose actual iteration count or a convergence
flag; those are explicitly unavailable, not fabricated from benchmark accuracy.

## Exact initialization and validated algorithms

The original weighted SO(3) PCA functions are called X then Y with the stored
ours_fit_seed RNG. Original BEST-improvement pairwise greedy examines all
three determinant-preserving flips per scan. R_init=(U_Y*greedy_signs)@U_X.T is
computed once and cached per pair, before any final refinement. The same
matrix starts MMD, ICP and CPD. Sigma=1; PCA max=200; final MMD max=100;
min gradient=1e-6; greedy cap=100 and improvement tolerance=1e-12. The MMD
refinement is exactly one native TrustRegions solve; no extra starts are used.
Its output reproduces the saved full Ours pipeline on every completed pair,
without separately rerunning the full adapter and recomputing initialization.

ICP calls the existing adapter with T_init[:3,:3]=R_init and zero translation,
correspondence distance 2, unchanged point-to-point estimation and Open3D
default convergence criteria (30 iterations, relative fitness/RMSE 1e-6).
It estimates translation, with scale fixed to one as before.

Installed pycpd 2.0.0's public R argument validates positive definiteness and
rejects some valid SO(3) matrices. Thus the documented fallback is used, without
patching the library: X_pre=Xc@R_init.T; the unchanged run_rigid_cpd adapter
fits X_pre to Yc at its default incremental identity, scale=1, translation=0.
Its R_inc is already R_library.T. Compose R_total=R_inc@R_init. For every fit,
s*(X_pre@R_inc.T)+t equals s*(Xc@R_total.T)+t numerically. CPD retains scale and
translation optimization, max_iterations=1000, tolerance=.001, w=0 and default
automatic initial variance. No diagnostic-history callback is added to timing.
Synthetic noncommuting-rotation tests verify the geometry independently of
EMDB accuracy; validation is performed before the one-pair smoke gate.

## Verified reuse and runtime accounting

Identity ICP/CPD and Go-ICP rows are reused only after the existing dataset and
checkpoint validators check the full 80-row reference against manifests,
input/rotation hashes, method settings, code, package versions and environment.
Their original runtimes and transformations are retained and labelled as prior
measurements. No new Go-ICP search is run. Go-ICP searches full SO(3) via the
axis-angle cube [-3.1416,3.1416]^3 enclosing the pi-radius rotation ball, and
translation [-.05,.05]^3, DT=300, MSE=.001, all 1000 points, trim=0, DT expansion=2.
The existing guarded rotation-cleanup policy (5e-6) remains unchanged. It has
no single local R_init, so its initial angle/refinement time are not applicable.
This global-search reference is not a claim of exact optimality under the
distance-transform approximation and finite stopping tolerance.

For shared methods, registration/refinement runtime is measured separately from
the common initialization. Per-method standalone total = initialization + its
refinement. The common time is shown in each standalone total but is counted
ONCE in the actual experiment ledger below, not summed three times. Reused
historical fit costs are zero new execution cost, yet remain visible in the
method runtime comparison. The boxes show total standalone cost; open circles
show shared refinement-only times, connected to their totals.

```json
{
  "initialization_computations": 20,
  "common_initialization_seconds_counted_once": 6.480728200171143,
  "new_registration_adapter_seconds": 239.6016692997655,
  "new_local_refinements": 80,
  "reused_identity_ICP_CPD_GoICP_rows": 60,
  "historical_runtime_not_added_to_new_execution_cost": true,
  "unique_init_plus_new_fit_seconds": 246.08239749993663
}
```

Identical initial rotations do not imply identical objectives or parameter
spaces: MMD is rotation-only; ICP/Go-ICP estimate translation; CPD also estimates
scale. The PCA/greedy initialization is selected using MMD and may favor it.
Identity-start MMD/ICP/CPD are the complementary starting-rotation control.
Convergence is separate from accuracy; no solver parameters were tuned using
true rotations and no local solver is claimed globally optimal.

## Outputs and preservation

trial_results.csv: 7 conditions/pair, initial/final angles, Frobenius errors,
improvement, split runtimes/provenance, translations/scales, convergence and
native diagnostics. initialization_diagnostics.csv stores each R_init, frames,
signs, initial MMD, initialization angle and PCA/greedy costs once per pair.
aggregated_results.csv, pairwise_results.csv and paired_comparison_summary.csv
provide descriptive summaries. validation_results.json, smoke_validation.json,
checkpoint_validation.json and preservation_check.json document validations.
All figures use actual SciencePlots ['science','no-latex'], 600-dpi PNG and
vector PDF. No old benchmark scripts, manifests, data, CSVs or figures changed.
The common-start computation is not a true-frame perturbation experiment.
No Overleaf files were edited.

Run/resume with the same validated Python environment:
```powershell
& .\data\.venv-emdb-baselines\Scripts\python.exe -B emdb_shared_initialization_comparison.py
& .\data\.venv-emdb-baselines\Scripts\python.exe -B emdb_shared_initialization_comparison.py --resume
```
The first completed pair is an incremental smoke gate before the remaining
pairs. Partial method rows and initialization records are atomic checkpoints;
resume verifies signatures and prefixes and skips completed work.

## Direct answers to the advisor's questions

1. Initialization is a major confound in the previous complete-pipeline comparison. Mean identity-to-shared error reductions are 110.606 degrees for MMD, 124.977 for ICP and 117.001 for CPD. This is evidence of strong initialization sensitivity, not a causal percentage attribution.
2. With the same start, mean final angles are ICP 1.267, CPD 1.440 and MMD 6.369 degrees. Both shared ICP and shared CPD beat shared MMD in all 20 paired comparisons. ICP beats CPD on 11 pairs; CPD beats ICP on 9. These are descriptive results for this frozen dataset only.
3. Starting from a mean angle of 6.622 degrees, mean angular improvements are 5.355 degrees for ICP, 5.182 for CPD and 0.253 for MMD. ICP/CPD improve every pair; MMD improves 10 and worsens 10.
4. Shared CPD substantially improves over identity CPD: mean error 118.441 -> 1.440 degrees. Its returned scale/translation remain free and recorded; correct noncommuting composition is independently validated.
5. All 10 shared-MMD angular deteriorations occur despite decreasing its empirical MMD. All 20 shared-MMD solves and all 20 identity-MMD solves satisfy the original gradient tolerance; all 40 CPD rows satisfy their original tolerance. The observations are not explained by a transpose/composition mistake or a reported tolerance failure. They are compatible with empirical-loss preference differing from angular accuracy; this experiment alone does not establish the statistical cause.
6. Go-ICP's verified historical mean angle is 1.390 degrees, with mean runtime 129.580 seconds. Shared ICP is lower on 12 pairs and Go-ICP on 8; shared CPD is lower on 7 pairs and Go-ICP on 13. Shared standalone mean runtimes including initialization are ICP 0.329, MMD 1.283, CPD 8.236 seconds. Go-ICP optimizes a DT-based distance objective with bounded translation and finite tolerance; a global-search reference need not have the lowest true angular error. Do not infer general superiority or exact global optimality.

## Execution and I/O recovery

A transient Windows destination-file lock stopped an atomic CSV replacement after 97 durable rows. The old valid checkpoint remained intact. A retry was added only in this new driver; AST comparisons verified that all fitting, evaluation and statistical functions were unchanged. All 97 committed rows and 14 saved initializations were reused on resume. No completed refinement was rerun. The only uncommitted row was a cached Go-ICP reference, which required no search. The full completed checkpoint was also tested in a temporary copy with fitting and initialization functions forbidden: all 140 rows were reused without any fit or initialization call. Original failure details and the I/O-only source migration proof are retained.

The runtime ledger covers both execution segments: unique common initialization 6.481 seconds plus new local adapters 239.602 seconds = 246.082 recorded algorithm seconds. This excludes preflight/tests, I/O, plotting and interruption handling. experiment_metadata.json's invocation_elapsed_seconds is the final resumed invocation's wall time, not the full interrupted experiment's wall time. Cached historical runtimes are not charged to new execution.
