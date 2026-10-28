# Held-out MMD cross-evaluation: EMD-9515, all 20 fixed rotation sets

Status: completed. 700/700 evaluation repetitions; five fixed rotations each.

No registration, initialization, rotation correction or bandwidth tuning. Optional Go-ICP omitted.

## Protocol

Reuse diagnose_emdb_heldout_mmd.sample_pair unchanged: threshold >0.035, 78351 eligible voxels,
coordinates (x,y,z)*voxel_size.x, uniform without replacement within each cloud, independent clouds
with overlap allowed; %.4f ASCII round-trip before rotation. Y=B@R_true.T; independently center A/Y
and divide both by one common maximum centered radius, once per repetition. This sample-dependent
centering/radius is shared by ALL candidates; no scale/translation from a fit is applied.

Each fitted pair/size/repetition has two distinct predeclared streams: SeedSequence([20261027,
1213149744, pair_index, n, repetition]).spawn(2), generate_state(1,uint64), default_rng.
The plan is saved before evaluation. Streams are distinct from training and legacy evaluation seeds.
Evaluation clouds are NOT reused across fitted pairs. Independent draws may overlap in voxels.

Exact float64 Gaussian kernel sigma=1; diagonal-excluding within terms computed once/cloud using
symmetric tiles; cross means computed with the same cdist/exp as RotationMMD.evaluate.
Losses are unscaled and may be negative. Delta=alternative minus MMD; within terms cancel in deltas.
Numerical preference tolerance: 1e-12; validation tolerance: 3e-14.

## Primary results

| Comparison | Mean prefers alternative | Mean prefers MMD | Training sign reversals | Clear alternative | Clear MMD | Uncertain | CI excludes zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICP - MMD | 20/20 | 0/20 | 20/20 | 17/20 | 0/20 | 3/20 | 17/20 |
| CPD - MMD | 20/20 | 0/20 | 20/20 | 17/20 | 0/20 | 3/20 | 17/20 |
| True - MMD | 20/20 | 0/20 | 20/20 | 18/20 | 0/20 | 2/20 | 18/20 |
| Initialization - MMD | 8/20 | 12/20 | 8/20 | 6/20 | 10/20 | 4/20 | 16/20 |

## Sample sizes (descriptive across fitted pairs)

| n | comparison | completed pairs | mean of pair means | median | range | alternative-preferred means | clear alternative CIs |
|---:|---|---:|---:|---:|---|---:|---:|
| 1000 | ICP - MMD | 20/20 | -1.841139493006e-05 | -1.461983547416e-05 | [-6.605563355253e-05, -3.429431792934e-06] | 20 | 17 |
| 1000 | CPD - MMD | 20/20 | -1.823709917395e-05 | -1.402401281059e-05 | [-6.542292609463e-05, -1.037634780687e-06] | 20 | 17 |
| 1000 | True - MMD | 20/20 | -1.949192497316e-05 | -1.574598003131e-05 | [-6.661511080670e-05, -3.289190530320e-06] | 20 | 18 |
| 1000 | Initialization - MMD | 20/20 | 1.956234712783e-06 | 2.033056194384e-06 | [-1.261250027643e-05, 2.409545747480e-05] | 8 | 6 |
| 5000 | ICP - MMD | 20/20 | -1.501546233451e-05 | -1.070467646282e-05 | [-4.770899473452e-05, -3.115008858456e-06] | 20 | 19 |
| 5000 | CPD - MMD | 20/20 | -1.473013931595e-05 | -1.080905335804e-05 | [-4.761670921671e-05, -2.790521347462e-06] | 20 | 17 |
| 5000 | True - MMD | 20/20 | -1.597886412724e-05 | -1.179020804675e-05 | [-4.888806501240e-05, -3.353576056764e-06] | 20 | 19 |
| 5000 | Initialization - MMD | 20/20 | 1.586649747045e-06 | 7.046959117907e-07 | [-7.461006554887e-06, 2.132813862685e-05] | 9 | 8 |
| 10000 | ICP - MMD | 20/20 | -1.341276557315e-05 | -1.124739718141e-05 | [-4.542713343851e-05, -1.635134744049e-06] | 20 | 18 |
| 10000 | CPD - MMD | 20/20 | -1.322048507275e-05 | -9.805596782253e-06 | [-4.431574185024e-05, -2.446199144313e-06] | 20 | 18 |
| 10000 | True - MMD | 20/20 | -1.421586555792e-05 | -1.139207002316e-05 | [-4.558996624775e-05, -2.428241627284e-06] | 20 | 18 |
| 10000 | Initialization - MMD | 20/20 | 1.203975766113e-06 | 3.767628133700e-07 | [-8.318363887527e-06, 1.636480741181e-05] | 9 | 7 |

## Pair_017

The legacy fixed R_mmd and R_true are bitwise identical to the new candidates. All 35 old seeds
were replayed separately; source/target XYZ hashes, normalized hashes, radius and overlap match.
Maximum absolute earlier Ours-True delta discrepancy: 4.4408920985006262e-16.
Legacy Ours-True is NEGATED to compare with the new True-MMD sign convention.

| n | comparison | training delta | held-out mean | pointwise 95% CI | reversal |
|---:|---|---:|---:|---|---|
| 1000 | ICP - MMD | 1.899716297027e-05 | -1.128968710850e-05 | [-2.039265321363e-05, -2.186721003364e-06] | True |
| 1000 | CPD - MMD | 2.412672121399e-05 | -1.282509098287e-05 | [-2.286325225172e-05, -2.786929714024e-06] | True |
| 1000 | True - MMD | 2.117447162120e-05 | -1.306273118898e-05 | [-2.133481995802e-05, -4.790642419948e-06] | True |
| 1000 | Initialization - MMD | 1.922623512840e-06 | 8.231327621033e-06 | [3.833894326329e-06, 1.262876091574e-05] | False |
| 5000 | ICP - MMD | 1.899716297027e-05 | -1.717983986083e-05 | [-2.135254889543e-05, -1.300713082622e-05] | True |
| 5000 | CPD - MMD | 2.412672121399e-05 | -1.694509269787e-05 | [-2.267540727842e-05, -1.121477811733e-05] | True |
| 5000 | True - MMD | 2.117447162120e-05 | -1.797296861883e-05 | [-2.308161346165e-05, -1.286432377601e-05] | True |
| 5000 | Initialization - MMD | 1.922623512840e-06 | 8.007327779502e-06 | [6.037756504004e-06, 9.976899055000e-06] | False |
| 10000 | ICP - MMD | 1.899716297027e-05 | -1.501770447057e-05 | [-2.404347269924e-05, -5.991936241888e-06] | True |
| 10000 | CPD - MMD | 2.412672121399e-05 | -1.492610115585e-05 | [-2.288121987094e-05, -6.970982440767e-06] | True |
| 10000 | True - MMD | 2.117447162120e-05 | -1.652098744018e-05 | [-2.569455219791e-05, -7.347422682460e-06] | True |
| 10000 | Initialization - MMD | 1.922623512840e-06 | 5.895964564262e-06 | [4.555286898165e-06, 7.236642230359e-06] | False |

## Interpretation and uncertainty

Training preference usually reverses on held-out clouds, supporting finite-sample training-loss overfitting/objective mismatch for these fixed fits.
Inspect the per-comparison counts and intervals above; mean signs alone do not establish a clear preference.
CIs use the paired difference from each whole independent evaluation repetition, conditional on the fixed fitted rotations.
Pointwise Student-t: mean +/- t_(r-1,.975)*sample_SD/sqrt(r); iid repetitions under the specified sampling/preprocessing law,
and approximately normal mean (exact coverage would require normal repetition differences). At r=5/10, normality is not
assured and interval coverage is approximate. No multiplicity adjustment; the CI counts are descriptive, not a familywise claim.
No candidate variances are combined as independent. No pooled repetition-level inference across fitted pairs is used.
These intervals are conditional Monte Carlo uncertainty, not uncertainty over a population of trained registrations.
Failure to exclude zero is not equivalence. Larger n also changes the random radius and centering, and repetitions decrease
with n; no rate or improved registration accuracy follows from these fixed-rotation evaluations.
This does not establish population optimality/consistency, local/global solver optimality or universal method superiority.
A next study could increase independent repetitions for unresolved pairs with the same predeclared protocol; do not tune using truth.

## Numerical checks, preservation and outputs

Dense/blockwise maximum discrepancy: 1.1102230246251565e-15.
Candidate source conditions, hashes, strict SO(3), CPD total rotation composition and original source inventories verified.
No candidates projected, transposed anew, inverted or otherwise corrected. All fit/TrustRegions/correction calls guarded.
Float64 accumulation uses math.fsum of tile sums. All five candidates share identical read-only evaluation arrays.
All earlier files are hash-protected. See preservation_check.json and output_manifest.json for exact paths/checksums.
Repetition CSV records all five absolute losses, four raw deltas, cross means, within sum, seeds, hashes and runtime.
Per-pair summaries contain fixed angular errors, sample SDs, CIs, repetition preferences and training sign changes.
All figures use actual SciencePlots science/no-latex, linear signed axes, 600-dpi PNG and vector PDF. Display-only x1e6.
Existing experiments and Overleaf are untouched.

## Completion by setting

- n=1000: 400/400 repetitions (20/pair planned); complete.
- n=5000: 200/200 repetitions (10/pair planned); complete.
- n=10000: 100/100 repetitions (5/pair planned); complete.

## Final verification

All 700 repetitions (3500 fixed-candidate evaluations) completed in 672.413 seconds. At every evaluation sample size, mean ICP-MMD, CPD-MMD and True-MMD differences are negative on all 20 pairs. Pointwise interval counts vary by sample size; increased n does not imply monotonically increasing certainty with fewer repetitions.

Seven evaluation-only tests passed. All 240 per-pair mean/SD/CI summaries recompute exactly from the CSV. All 398 protected files are unchanged. A completed-run resume reused all 700 checkpoint rows, performed only one deterministic validation replay, and left the repetition CSV byte-for-byte unchanged. All six figure exports passed DPI/vector/font checks and visual inspection. See verification_audit.json and resume_validation.json.
