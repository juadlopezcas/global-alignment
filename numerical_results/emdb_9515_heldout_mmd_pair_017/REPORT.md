# Held-out MMD validation: EMD-9515 pair_017

## Fixed training comparison and candidates

Training MMD(Ours)-MMD(True) = -2.1174471621199231e-05: the training objective
preferred Ours. The exact previously saved R_final, R_true and proper R_goicp
matrices are frozen; no candidate was fitted/refined and sigma stays 1.
The Go-ICP proper matrix reuses the prior explicit SO(3) projection, distance
2.8661159715603498e-07; the raw matrix and policy
are preserved in fixed_candidates.json. No transpose/inverse or translation is
used. The angular errors remain approximately 13.4114 degrees (Ours) and
0.448785 degrees (proper Go-ICP; saved raw benchmark angle about 0.449652).

## Held-out results

| n | repetitions | mean Ours - True | sample SD | 95% mean CI | fraction > 0 | interpretation |
|---:|---:|---:|---:|---|---:|---|
| 1000 | 20 | 2.380109361714e-05 | 1.437059667493e-05 | [1.707544734442e-05, 3.052673988986e-05] | 0.950 | True preferred |
| 5000 | 10 | 1.855054791100e-05 | 8.493066488951e-06 | [1.247497414541e-05, 2.462612167659e-05] | 1.000 | True preferred |
| 10000 | 5 | 1.293415204788e-05 | 5.009427378453e-06 | [6.714126420442e-06, 1.915417767532e-05] | 1.000 | True preferred |

Positive Delta(a,b) means b has lower MMD. All three paired contrasts:

| n | comparison | mean delta | 95% mean CI | fraction > 0 | interpretation |
|---:|---|---:|---|---:|---|
| 1000 | Delta_ours_true | 2.380109361714e-05 | [1.707544734442e-05, 3.052673988986e-05] | 0.950 | second candidate preferred |
| 1000 | Delta_goicp_true | 6.683772191263e-07 | [-5.916046162453e-07, 1.928359054498e-06] | 0.650 | CI includes zero; not clearly distinguished |
| 1000 | Delta_ours_goicp | 2.313271639801e-05 | [1.684617961932e-05, 2.941925317670e-05] | 1.000 | second candidate preferred |
| 5000 | Delta_ours_true | 1.855054791100e-05 | [1.247497414541e-05, 2.462612167659e-05] | 1.000 | second candidate preferred |
| 5000 | Delta_goicp_true | -1.640421155580e-07 | [-1.078531109468e-06, 7.504468783523e-07] | 0.400 | CI includes zero; not clearly distinguished |
| 5000 | Delta_ours_goicp | 1.871459002656e-05 | [1.200976270158e-05, 2.541941735154e-05] | 1.000 | second candidate preferred |
| 10000 | Delta_ours_true | 1.293415204788e-05 | [6.714126420442e-06, 1.915417767532e-05] | 1.000 | second candidate preferred |
| 10000 | Delta_goicp_true | 4.855783295454e-08 | [-8.289288988151e-07, 9.260445647242e-07] | 0.800 | CI includes zero; not clearly distinguished |
| 10000 | Delta_ours_goicp | 1.288559421493e-05 | [6.215931146858e-06, 1.955525728300e-05] | 1.000 | second candidate preferred |

The aggregate CSV additionally records median/min/max/sample SD. Each sample
size has its own marginal 95% Student-t interval:
mean +/- t_(r-1,0.975) * sample_SD/sqrt(r). The independent uncertainty unit is
the entire resampling repetition, not individual points, kernel pairs or tiles.
No sizes are pooled. These intervals assume independent repetitions and an
approximately normal sampling distribution of their mean; r=5 gives limited
precision. Intervals containing zero do not clearly distinguish candidates at
that size, and do not prove equivalence. The intervals are not simultaneous
multiple-comparison confidence guarantees.

## Sampling, preprocessing and exact objective

All planned seeds were stored in sampling_plan.csv before objective evaluation.
SeedSequence([master_seed, 0x48454C44, n, repetition]).spawn(2), uint64 child
integer seeds and separate default_rng streams. All 70 declared seeds are
distinct and differ from all 80 original training stream seeds. Master seed:
20261007. Stopping for runtime never silently changes repetitions.

Population: data/emd_9515.map, density > 0.035, 78351 voxels;
z,y,x=np.where(...), column_stack((x,y,z))*voxel_size.x. Uniform without
replacement within each independently drawn X/Z. No density weights, noise,
interpolation, origin/axis corrections, exclusions of training voxels or forced
disjointness. Coordinate values use the exact existing %.4f ASCII round-trip.
Per-repetition seeds, point counts, overlaps, XYZ/normalized-array hashes and
normalization scales are in heldout_trial_results.csv.

Y_eval=Z_eval@R_true.T; existing center_and_normalize is called once per pair.
All fixed candidates then share the same read-only Xc/Yc. Ground truth only
generates this evaluation target and defines a fixed comparison candidate; it
never chooses bandwidth, seeds, normalization or a fitted rotation.

The unchanged diagonal-excluding MMD within-cloud terms cancel under proper
rotations. Differences use -2*(mean_cross_kernel(a)-mean_cross_kernel(b)),
with exp(-squared_distance/(2*sigma^2)), sigma=1. Both arrays are tiled at
chunk_size=512; float64 tile sums plus math.fsum are exact
all-pairs evaluation up to floating-point arithmetic. No full 10000x10000
matrix, random features or approximate kernel is used. Only displayed axes
are multiplied by 1e6. Block distance algebra matches RotationMMD.problem;
direct original mmd_squared checks use the original cdist kernel.

## Numerical validation and preservation

Validation passed at n=1000 before larger stages. Maximum cross-only vs full
original MMD discrepancy: 8.8817841970012523e-16; maximum chunk-size difference
discrepancy across 257/512/1024 and configured size: 1.1102230246251565e-15.
Saved-seed regeneration is byte/array exact, identical candidates give exact
zero difference, and row/column transformation conventions agree. Unit tests
also check shared evaluation samples, no normalization per candidate, no
fitting calls, unchanged rotation bytes and original-file hashes.

Numerical discrepancies at this scale are distinct from the across-repetition
sampling SD. CI widths estimate sampling variability, not numerical error.
protected_file_hashes.json/preservation_check.json protect pre-existing code,
frozen data and all completed numerical results. No new registrations or
sample-size scaling of fitted rotations were performed.

## Stability and computational cost

Observed Ours-vs-True sample SD by size: n=1000: 1.437060e-05 -> n=5000: 8.493066e-06 -> n=10000: 5.009427e-06.
The observed repetition-level SD decreases at each larger evaluation size, so the loss difference becomes more stable in these draws. Mean-CI widths do not shrink monotonically because larger evaluations use fewer repetitions and larger Student-t critical values.
Different sample sizes use independent streams; counts 20/10/5 limit how
precisely a monotone SD trend can be established. Shared normalization uses
each pair's maximum centered radius, so increasing n can also alter the scale
distribution/effective geometry; its scale range is recorded, not held fixed
or tuned. Scale ranges: [{"n": 1000, "min": 106.64622408613631, "max": 131.0426404914734}, {"n": 5000, "min": 110.99024570305419, "max": 131.60009520201038}, {"n": 10000, "min": 111.79130972940843, "max": 131.59047124729366}]. Describe these results
rather than claiming a variance law. The distribution figure shows empirical
cumulative distributions, not a fitted density estimate.

CPU NumPy backend; recorded numeric diagnostic invocation elapsed
46.280 seconds. Per-stage times: [{"n": 1000, "repetitions": 20, "kernel_seconds": 1.0109209001529962, "pair_total_seconds": 1.1451593001838773}, {"n": 5000, "repetitions": 10, "kernel_seconds": 12.70196499954909, "pair_total_seconds": 13.063307100441307}, {"n": 10000, "repetitions": 5, "kernel_seconds": 28.392139800125733, "pair_total_seconds": 28.729486900148913}].
Pre-larger-stage quadratic estimates: [{"num_points": 5000, "planned_repetitions": 10, "predicted_kernel_seconds": 12.758012511767447, "basis": "median measured n=1000 three-candidate kernel time * (n/1000)^2 * repetitions"}, {"num_points": 10000, "planned_repetitions": 5, "predicted_kernel_seconds": 25.516025023534894, "basis": "median measured n=1000 three-candidate kernel time * (n/1000)^2 * repetitions"}].
Runtime stop reason: none; all 35 predeclared repetitions completed.
Only tile-sized matrices (at most 512x512) are
allocated by the blocked evaluator; multiple temporary tiles may coexist.
Software versions, environment, source hashes, plan and timing scope are in
run_metadata.json. Trial runtime includes sampling/serialization, one shared
preprocessing step and all three candidate evaluations; kernel runtime is
recorded separately. Validation, plots and I/O are excluded from trial timers.

## Interpretation and next experiment

Held-out results support A for the evaluated sizes: fixed R_true has lower MMD than the fitted training rotation, reversing the training preference. This supports a finite-sample estimation error explanation for this pair. It is not a population-consistency proof.

A subsequent controlled sample-size scaling study of fitted rotations is
scientifically justified to test whether training-sample error decreases with
larger independently resampled training clouds. It should use predeclared
training sizes/seeds, the unchanged sigma and algorithm, and independent
held-out evaluation. If the present intervals remain unresolved, expand
evaluation repetitions first. The present experiment freezes one training
pair's rotations and does not itself measure the effect of larger training
samples or establish performance across pairs/maps. No such registrations are
run here, and sigma was not chosen using R_true.
