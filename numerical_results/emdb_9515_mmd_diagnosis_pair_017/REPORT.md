# MMD landscape diagnosis: EMD-9515 independent-resampling pair_017

This is a single-pair research diagnostic, not a benchmark or algorithm change.
All fitting here uses sigma=1. Go-ICP/ICP/CPD were NOT rerun. Saved Go-ICP
translation is excluded from every rotation-only MMD comparison.

## Exact reproduction

The original `emdb_baseline_comparison.run_ours` adapter was called with frozen
X/Z, stored R_true and `default_rng(ours_fit_seed)` after the unchanged
`emdb_independent_resampling_comparison.prepare_pair` preprocessing. Reproduction
passed before any diagnostic sweeps: rotation Frobenius difference
0.0000000000000000e+00, angular-error difference
0.0000000000000000e+00, final MMD difference
0.0000000000000000e+00. Angle is
13.411418920038 degrees; iterations
1; gradient 9.3515656055468729e-07.
Both geometric frames and the greedy rotation are returned by the unchanged
registration implementation, so the intermediate initializations are recovered
unambiguously. `R_pca_initial=U_Y@U_X.T`; greedy signs
[-1, -1, 1]; `R_greedy_initial=(U_Y*signs)@U_X.T`.
Matrices and reproduction details are in reproduction_check.json/rotations.json.

The saved Go-ICP matrix has orthogonality residual
5.7322311426322117e-07; determinant
9.9999976292781356e-01. For exact SO(3) geodesics/tangent
analysis and external-start refinement only, an explicit nearest-proper-SO(3)
SVD projection changes it by 2.8661159715603498e-07
in Frobenius norm. The saved raw matrix is preserved in rotations.json and its
rotation-only MMD is -3.7316812118071141e-04.
The primary candidate table uses the proper rotation; no benchmark CSV or
Go-ICP cleanup policy was changed. No transpose/inverse was applied.
The additional R_goicp_saved_raw row also reports the exact stored matrix's
MMD, benchmark angle and implemented Pymanopt projected-gradient norm. Because
that raw matrix is slightly outside exact SO(3), its gradient norm is an
approximate numerical diagnostic, not an exact intrinsic gradient at an SO(3)
point. The saved raw benchmark angle is
0.449651599470 degrees, versus
0.448785370828 degrees after geometry-only cleanup.
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

| candidate | mmd | mmd_minus_final | gradient_norm | angular_error_degrees | distance_from_final_degrees |
| --- | --- | --- | --- | --- | --- |
| R_pca_initial | -5.031007530087e-05 | 3.437006714055e-04 | 1.578941817067e-04 | 1.787896767510e+02 | 1.787410436679e+02 |
| R_greedy_initial | -3.920881231936e-04 | 1.922623512840e-06 | 9.589461374056e-05 | 1.533003483925e+01 | 2.254663268025e+00 |
| R_final | -3.940107467064e-04 | 0.000000000000e+00 | 9.351565605547e-07 | 1.341141892004e+01 | 0.000000000000e+00 |
| R_true | -3.728362750852e-04 | 2.117447162120e-05 | 1.291723450854e-04 | 1.707547292503e-06 | 1.341141892004e+01 |
| R_goicp | -3.731553173478e-04 | 2.085542935859e-05 | 1.296202117981e-04 | 4.487853708283e-01 | 1.327713237281e+01 |
| R_goicp_saved_raw | -3.731681211807e-04 | 2.084262552571e-05 | 1.296197418284e-04 | 4.496515994695e-01 | 1.327716104826e+01 |

## Gradient finite-difference validation

Central differences in double precision used three orthogonal axes (first toward
truth), at h=0.01, 0.001, 0.0001 and 0.00001 radians for greedy, final and truth.
All 36 absolute/relative discrepancies are in gradient_checks.csv. For the
requested practical h=1e-3/1e-4, maximum absolute discrepancy is
1.6250930997749585e-10; maximum relative
discrepancy is 5.9251533705424770e-05.
The derivative checks passed; no gradient formula was changed. Relative errors
near zero derivatives can be sensitive to cancellation; raw derivatives and
absolute discrepancies remain available.

## Geodesics and local landscape

Paths use Exp(t*Log(R_endpoint@R_final.T))@R_final at 81 points on
[-0.25,1.25]. This sampling does not hit t=0 or t=1 exactly; endpoint values
were evaluated separately. Barriers below are computed on [0,1], with exact
endpoints included. Values are sampled evidence, not certified extrema.

| path | start_mmd | endpoint_mmd | minimum_mmd | minimum_t | barrier_above_start_on_segment | barrier_above_higher_endpoint | lower_region_away_from_start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truth | -3.940107467064e-04 | -3.728362750852e-04 | -3.940102316990e-04 | -6.250000000000e-03 | 2.117447162120e-05 | 0.000000000000e+00 | False |
| goicp | -3.940107467064e-04 | -3.731553173478e-04 | -3.940102168187e-04 | -6.250000000000e-03 | 2.085542935859e-05 | 0.000000000000e+00 | False |

Extended [-0.25,1.25] sampled maxima/barriers are also recorded:

| path | maximum_mmd_sampled_full_range | barrier_above_start_full_range |
| --- | --- | --- |
| truth | -3.612732416216e-04 | 3.273750508481e-05 |
| goicp | -3.617580342301e-04 | 3.225271247631e-05 |

The 21x21 local grid uses Exp(skew(a*u+b*v))@R_final with a,b in +/-20
degrees. Its coordinates are recorded in local_landscape.csv; truth is marked
at its actual in-plane coordinates. Go-ICP out-of-plane residual is
3.4554462603328395e-03 radians and marked
on the contour: False. It is never projected
onto the displayed tangent plane merely for plotting. Plot y-values/color are
(L-L(R_final))*1e6 for readability only; solver/objective are not rescaled.

Directional curvatures (per radian squared) for three Euclidean unit axes:

| direction | h_radians | curvature_per_radian_squared |
| --- | --- | --- |
| 0 | 1.000000000000e-03 | 7.842479998743e-04 |
| 0 | 1.000000000000e-04 | 7.842837490557e-04 |
| 1 | 1.000000000000e-03 | 6.632547178143e-03 |
| 1 | 1.000000000000e-04 | 6.632649984795e-03 |
| 2 | 1.000000000000e-03 | 9.226108099725e-03 |
| 2 | 1.000000000000e-04 | 9.226064356938e-03 |

The finite-difference full 3x3 local Hessian at h=1e-3 has eigenvalues
[0.000779635857719487, 0.005822815047429227, 0.01004045237259282] in rotation-vector coordinates. For metric-unit
axes these eigenvalues divide by 2. Curvature checks across step sizes are in
directional_curvatures.csv. A slice, positive directional curvatures or a local
positive-definite Hessian cannot establish a global optimum.

## Final-gradient threshold experiment

PCA and greedy are frozen. All three final TrustRegions runs start at exactly
the recovered greedy matrix; only min_gradient_norm varies. max_iterations=100,
use_rand=False, and every other default remain unchanged. Complete stopping
messages/runtime and matrices are in optimizer_threshold_comparison.csv.

| min_gradient_norm | initial_mmd | final_mmd | initial_gradient_norm | final_gradient_norm | iterations | displacement_degrees | angular_error_degrees | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000000000000e-06 | -3.920881231936e-04 | -3.940107467064e-04 | 9.589461374056e-05 | 9.351565605547e-07 | 1 | 2.254663268025e+00 | 1.341141892004e+01 | 6.737693999894e-01 |
| 1.000000000000e-08 | -3.920881231936e-04 | -3.940109291056e-04 | 9.589461374056e-05 | 1.060263112151e-10 | 2 | 2.246835143867e+00 | 1.342963668481e+01 | 1.151551699964e+00 |
| 1.000000000000e-10 | -3.920881231936e-04 | -3.940109291058e-04 | 9.589461374056e-05 | 1.471961680016e-17 | 3 | 2.246837481358e+00 | 1.342963482110e+01 | 1.807534700027e+00 |

## ORACLE/EXTERNAL diagnostic starts

These runs are not legitimate initializations for the proposed method and
never enter benchmark results. Each uses the same final TrustRegions settings
with min_gradient_norm=1e-10; no oracle PCA or sign search is performed.

| label | min_gradient_norm | initial_mmd | final_mmd | final_gradient_norm | iterations | displacement_degrees | angular_error_degrees | stopping_criterion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R_goicp_EXTERNAL | 1.000000000000e-10 | -3.731553173478e-04 | -3.940109291056e-04 | 9.041279342910e-13 | 3 | 1.329514479261e+01 | 1.342963481683e+01 | Terminated - min grad norm reached after 3 iterations, 2.04 seconds. |
| R_true_ORACLE | 1.000000000000e-10 | -3.728362750852e-04 | -3.940109291054e-04 | 9.919961044238e-13 | 3 | 1.342963481437e+01 | 1.342963481437e+01 | Terminated - min grad norm reached after 3 iterations, 1.86 seconds. |

## Evidence and interpretation

- Early stopping effect: the strict greedy-start run decreases the saved final
  MMD by 1.8239942889408667e-10, moves
  2.246837481358 degrees from initialization, and has
  13.429634821096 degrees error. Check its final gradient
  and stopping criterion, rather than assuming its iteration count means success.
- Local-basin evidence: best external/oracle-start final MMD minus strict
  greedy-start MMD is 2.2204460492503131e-16; compare
  final rotations and error, not native Go-ICP SSE. No global optimality claim
  follows from these two external starts.
- Empirical-objective preference: truth minus saved Ours MMD is
  2.1174471621199231e-05; proper Go-ICP minus saved Ours MMD is
  2.0855429358590527e-05. A lower MMD need not recover the imposed
  population rotation better on two independently resampled finite clouds.
- Flatness/identifiability: inspect the small directional curvatures, Hessian
  spectrum and plotted loss changes against double-precision derivative checks.
  Numerical flatness is different from proven statistical uncertainty; this one
  pair provides no sampling confidence intervals.

## Dominant limitation and next scientific experiment

The dominant observed limitation is empirical-objective mismatch (C).
The stricter solve, truth start and external Go-ICP start converge to essentially
the same rotation (largest external-vs-strict Frobenius difference
3.4000967242656930e-10) at about 13.429634821 degrees
from the imposed truth. Starting at truth actually moves away from it while
lowering MMD. The truth/Go-ICP orientations have higher MMD by approximately
2.1e-5, so these checks provide no evidence of a better-MMD basin near truth (B).

Early termination (A) has only a minor effect here: the stricter solution moves
0.022357158200 degrees from the saved final rotation and lowers MMD
by 1.8239942889408667e-10, only 8.61411950e-06 of the truth-vs-final loss
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
