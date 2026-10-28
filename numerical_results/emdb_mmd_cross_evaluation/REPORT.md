# Rotation-only MMD cross-evaluation: EMD-9515

Evaluation only: 20 existing frozen pairs, 1000 points/cloud, five exact saved
candidate matrices/pair, sigma=1. No registration, TrustRegions, sampling,
candidate refinement, new normalization or hyperparameter selection occurred.
Go-ICP is optional and deliberately omitted; the required five-candidate
comparison isolates the shared-start refinements without an extra near-SO(3)
native-rotation representation issue. No Go-ICP search or new cleanup was used.

## Observed loss preferences

| Candidate | Strict lowest loss among MMD/ICP/CPD | Lowest/tied within tolerance |
|---|---:|---:|
| R_mmd | 20/20 | 20/20 |
| R_icp | 0/20 | 0/20 |
| R_cpd | 0/20 | 0/20 |

| Difference | Mean | Median | Minimum | Maximum | MMD lower / alternative lower / ties |
|---|---:|---:|---:|---:|---|
| ICP_minus_MMD | 1.561213263317e-05 | 1.354460837288e-05 | 3.061366219193e-06 | 5.544505826593e-05 | 20 / 0 / 0 |
| CPD_minus_MMD | 1.803780902628e-05 | 1.258784739400e-05 | 2.788504308571e-06 | 7.348777475880e-05 | 20 / 0 / 0 |
| TRUE_minus_MMD | 1.835302031904e-05 | 1.366504967193e-05 | 4.052096158169e-06 | 5.685006387313e-05 | 20 / 0 / 0 |
| MMD_minus_init | -6.675664772526e-07 | -3.520887454922e-07 | -2.693764141526e-06 | -7.472627960858e-08 | 20 / 0 / 0 |

Raw differences are unmodified, in original MMD units. ICP/CPD/TRUE minus MMD
positive means MMD has lower loss. MMD minus initialization negative means the
MMD refinement decreases its own loss. Classification uses the predeclared
absolute tolerance 1.0e-12: |difference|<=tolerance is a numerical
tie, not a statement of statistical equivalence. All raw signs and exact
values remain in the CSVs, including negative MMD estimates. The tolerance is
an arithmetic comparison convention, not a practical/significance threshold
and was not selected using angular errors. Angular accuracy comparisons use
1.0e-08 degrees to avoid classifying rounding-only changes.

## Answers to A-D

A. MMD decreases its empirical objective relative to initialization in
20/20 pairs; initialization is
lower in 0/20; numerical
ties 0/20. Mean MMD-minus-init
change is -6.6756647725263458e-07.

B. ICP has lower empirical MMD than the MMD result in
0/20 pairs; CPD in
0/20. Such a lower
observed candidate would be evidence that the saved MMD result is not the
lowest attainable among these candidates, not a proof of any global minimum.

C. MMD has lower loss but HIGHER angular error than ICP in
20/20 pairs and
than CPD in 20/20.
These counts compare the SAME rotation-only objective on the SAME arrays;
they are not comparisons of ICP RMSE, CPD likelihood or different objectives.

D. MMD is lower than the imposed truth in
20/20 pairs; truth is lower in
0/20, with
0/20 numerical ties. These are
empirical preferences on the original finite samples, not population MMD claims.

## Pair_017 and the earlier landscape diagnosis

| Candidate on pair_017 | Empirical MMD | Angular error (deg) | MMD difference from R_mmd |
|---|---:|---:|---:|
| R_init | -3.9208812319357911e-04 | 15.330034839246 | 1.9226235128400049e-06 |
| R_mmd | -3.9401074670641911e-04 | 13.411418920038 | 0.0000000000000000e+00 |
| R_icp | -3.7501358373615012e-04 | 1.513480344379 | 1.8997162970268988e-05 |
| R_cpd | -3.6988402549242849e-04 | 1.090141325853 | 2.4126721213990621e-05 |
| R_true | -3.7283627508521988e-04 | 0.000001707547 | 2.1174471621199231e-05 |

The saved R_init/R_mmd/R_true matrices and losses agree with the prior
landscape diagnosis (see validation_results.json). Shared ICP/CPD are the
NEW shared-start rotations, not the earlier identity baselines or Go-ICP
rotation. The existing trace/arccos angular metric can report a microdegree
rounding residual for truth versus itself; its raw value is retained.

## Numerical validation and provenance

The original manifest, source files, imposed rotations, core/generator/driver
code hashes, old baseline reference and shared-start output inventory hashes
are verified. The existing frozen-data and checkpoint validators run with all
fit/sampling entry points guarded against calls. The unchanged prepare_pair
is cached so normalization happens exactly once per pair in this diagnostic;
all candidates share those exact read-only Xc/Yc objects and input hashes.
No rotations are regenerated; R_init is loaded from its saved matrix file.

The original core.RotationMMD(Xc,Yc,sigma=1).evaluate(R) is used directly.
The diagonal-excluding within-cloud terms and original cross-kernel mean are
unchanged; negative values are valid. For ALL 100 candidates, independent
core.mmd_squared(Xc@R.T,Yc,1) agrees: maximum absolute discrepancy
2.6645352591003757e-15. Duplicate evaluations are
bitwise identical. Saved initial/final MMD and saved final angles agree with
the recomputation; maximum saved final-MMD discrepancy is
0.0000000000000000e+00. The independent check
tolerance is 3.0e-14, distinct from the loss-ranking tolerance.

All five matrices/pair pass strict SO(3) checks without any new projection.
ICP uses the saved shared total rotation. CPD uses the saved total rotation,
R_cpd=R_incremental@R_init, where the old adapter already transposed its internal
row rotation. Coordinate equivalence of the two-stage and composed rotations
is checked; maximum residual 3.3306690738754696e-16.
CPD scale and translation are inspected for provenance/composition only and
NEVER applied when evaluating MMD. ICP translation is likewise not applied.
The loss-evaluation function accepts only point clouds and rotations, no
candidate-specific translation, scale or preprocessing parameters.

The smallest absolute alternative-minus-MMD difference is
2.7885043085706940e-06. Compare this to
the observed arithmetic discrepancies above. Loss gaps are also reported
relative to numerical tolerances without interpreting a small number as a
sampling confidence interval. No point pairs are treated as independent
statistical observations; this report provides descriptive paired summaries.

## Interpretation, limitations and next experiment

The observed comparisons better support empirical MMD objective mismatch; none of the observed ICP/CPD candidates provides evidence of a lower MMD basin.

Lower angular error is not the optimized MMD criterion. The MMD result need
not be a true local/global minimum merely because it beats these finite
candidates or meets a saved gradient threshold. Conversely, an observed lower
candidate alone would not establish a reachable basin from the same start.
Objectives and transformation flexibility of the original solvers still
differ; ignoring their fitted scale/translation here isolates rotation under
one common evaluation loss but does not make their training objectives equal.

The most informative next step for objective mismatch is a predeclared held-out
cross-evaluation of these SAME fixed shared-start ICP/CPD/MMD rotations over
independent evaluation resamples, at unchanged sigma=1, with repetition-level
uncertainty. The earlier held-out pair_017 diagnostic used Ours/truth/Go-ICP;
it does not substitute for this broader fixed-candidate, 20-pair study. This
recommendation is not executed here. If any alternative has lower training
MMD, separately test tighter refinement or starts at that observed alternative
in a new, explicitly authorized optimization study. Do not tune using R_true.
No population-consistency or universal-method-superiority claim follows.

## Files and figures

candidate_mmd_values.csv has 100 candidate rows with full precision losses,
angles, matrices and input/rotation hashes. pairwise_mmd_differences.csv has
20 paired rows, raw differences and classifications. aggregated_diagnostics.csv
contains means, sample SDs, medians/ranges and preference counts;
lowest_loss_counts.csv distinguishes strict winners from tolerance ties.
validation_results.json and output_manifest.json contain the complete audit
and all output paths. All previous source/data/result files are hash protected.

All three figures use SciencePlots ['science','no-latex'], 600-dpi PNG and
vector PDF. Signed loss-difference axes are LINEAR. Differences are displayed
multiplied by 1e6, never rescaled during objective evaluation. The 20-panel
loss-vs-angle plot uses ACTUAL empirical losses multiplied by 1e6, independent
y ranges per pair, and common angular x range; it avoids pooling incompatible
pair-specific loss offsets into one misleading scatter. Exact data remain in
the CSVs. Elapsed diagnostic time before final reporting/audit: 21.041s.
No Overleaf files were changed.
