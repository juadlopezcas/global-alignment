# SciencePlots simulation: dimensions 2, 4, 8, 16

This is a separate 240-trial experiment. The runner imports the existing
`numerical_experiments.run_trial` and `aggregate_results` without copying or
changing registration, sampling, metrics or convergence settings.

## Configuration and scientific definitions

```json
{
  "dimensions": [
    2,
    4,
    8,
    16
  ],
  "n": 200,
  "num_trials": 20,
  "seed": 42,
  "noise_std": 0.05,
  "sigma": 1.0,
  "max_greedy_moves": 100,
  "greedy_tolerance": 1e-12,
  "max_iterations": 100,
  "pca_max_iterations": 200,
  "min_gradient_norm": 1e-06,
  "dataset_types": [
    "gaussian_ellipsoid",
    "uniform_box",
    "laplace_product"
  ]
}
```

Rows are points: Y=X@R_true.T+epsilon, epsilon iid N(0,0.05^2). The same 200
points underlie X and Y; there is no independent resampling. Gaussian,
Uniform(-sqrt(3)*axis_std,+sqrt(3)*axis_std) and Laplace(scale=axis_std/sqrt(2))
sources all have population covariance diag(d,d-1,...,1). Finite-sample
covariances need not equal population covariance. Each cloud is centered and
both share the maximum centered radius as a single normalization denominator.
The source sampler, noise draws, QR rotation and SeedSequence data/fit streams
are the original functions, keyed by seed/dimension/trial/dataset identity.
The dimension list does not change a matching trial's stream.

Two weighted SO(d) PCA solves use -trace(W@U.T@(data.T@data)@U), W=diag(d,...,1).
Every greedy step scores ALL d*(d-1)/2 pair flips, accepts the best improving
pair, then restarts a complete scan. A complete non-improving scan terminates
normally; 100 accepted moves is only a safety cap. One final TrustRegions
MMD solve starts at (U_Y*greedy_signs)@U_X.T. No first-improvement, extra starts
or ICP/CPD comparisons are used. Ground truth only enters evaluation.

The original frame error remains min_S ||R_true@U_X-U_Y@S||_F, over unrestricted
signed permutations matched by a dxd assignment; normalized frame error divides
by sqrt(d). Rotation error remains ||R_true-R_est||_F without signed-permutation
equivalence; its normalized value also divides by sqrt(d). Every existing
auxiliary metric/diagnostic is retained, including MMD, runtime, greedy work
and optimizer convergence. MMD and runtimes are not dimension-normalized.

## Main numerical results

| Distribution | d | Raw frame error: mean +/- sample SD | Normalized frame error: mean +/- SD |
|---|---:|---:|---:|
| Gaussian ellipsoid | 2 | 0.0050447544 +/- 0.0036043236 | 0.00356718 +/- 0.0025486417 |
| Gaussian ellipsoid | 4 | 0.022294165 +/- 0.01335594 | 0.011147082 +/- 0.0066779699 |
| Gaussian ellipsoid | 8 | 0.10866669 +/- 0.21952812 | 0.038419477 +/- 0.077614912 |
| Gaussian ellipsoid | 16 | 0.12081806 +/- 0.034444846 | 0.030204514 +/- 0.0086112116 |
| Uniform box | 2 | 0.006157007 +/- 0.005523433 | 0.0043536614 +/- 0.0039056569 |
| Uniform box | 4 | 0.020689766 +/- 0.01292157 | 0.010344883 +/- 0.0064607852 |
| Uniform box | 8 | 0.049774396 +/- 0.020540328 | 0.017597907 +/- 0.0072621025 |
| Uniform box | 16 | 0.10760688 +/- 0.028663022 | 0.026901721 +/- 0.0071657556 |
| Laplace product | 2 | 0.009234773 +/- 0.0062022566 | 0.0065299706 +/- 0.0043856577 |
| Laplace product | 4 | 0.022172374 +/- 0.0097771627 | 0.011086187 +/- 0.0048885813 |
| Laplace product | 8 | 0.068780171 +/- 0.044587799 | 0.024317463 +/- 0.015764168 |
| Laplace product | 16 | 0.11520545 +/- 0.029501285 | 0.028801362 +/- 0.0073753212 |

- Gaussian ellipsoid: raw mean increases at each tested dimension, from 0.0050447544 at d=2 to 0.12081806 at d=16. Normalized mean is not monotone across the tested dimensions (0.00356718 to 0.030204514).
- Uniform box: raw mean increases at each tested dimension, from 0.006157007 at d=2 to 0.10760688 at d=16. Normalized mean increases at each tested dimension (0.0043536614 to 0.026901721).
- Laplace product: raw mean increases at each tested dimension, from 0.009234773 at d=2 to 0.11520545 at d=16. Normalized mean increases at each tested dimension (0.0065299706 to 0.028801362).
- Lowest observed raw mean by dimension: d=2: Gaussian ellipsoid; d=4: Uniform box; d=8: Uniform box; d=16: Uniform box. This descriptive ranking is not a significance test.
- Largest individual frame error is 1.00000889 (gaussian_ellipsoid, d=8, zero-based trial=2); its registration rotation error is 0.0287080744. Wide frame-error SD bars do not imply the same magnitude of physical rotation error. All trials, including this one, remain in the statistics.
- Gaussian d=8 has raw SD greater than its raw mean. Its raw and normalized log plots therefore flag the omitted nonpositive lower SD whisker; both linear companions retain the complete interval. The Gaussian normalized mean decreases from d=8 to d=16, so normalized frame error is not universally monotone.

These compare distributions at fixed n/noise/sigma and each dimension. Raw
Frobenius error sums squared axis errors before taking the square root; the
normalized companion helps separate this dimension scaling from average axis
error. Frame errors describe estimated geometric frames, not final registration
rotation recovery. A monotone or distribution ordering should only be described
where the table supports it; these 20-trial summaries are not significance tests.

## Figures and plotting policy

All six figures use the actual SciencePlots 2.2.2
package with ["science", "no-latex"], registered by importing scienceplots.
See https://github.com/garrettj403/SciencePlots . No external LaTeX is used.
TrueType fonts are embedded in PDFs. PNGs use 600 dpi; PDFs contain vector
lines/text. Markers o/s/^ and line styles solid/dashed/dash-dot distinguish the
three distributions. X positions are numeric 2,4,8,16 on a LINEAR x-axis with
all four labels; the main raw/normalized frame figures have base-10 LOG y-axes
explicitly labeled '(log scale)'. Error bars are arithmetic mean +/- one sample
SD (ddof=1), not standard errors or confidence intervals.

If mean-SD<=0, a lower SD endpoint cannot be placed on a log axis. Its lower
whisker is omitted, its positive mean->mean+SD whisker retained, and a hollow
down-triangle at the mean plus a legend entry identifies the omission. No
positive floor is substituted and CSV statistics are never changed. Linear
raw/normalized companion plots always show the full mean +/- SD, including any
negative lower statistical bounds. Such bounds do not imply negative observed
errors. Omitted log lower whiskers: 2 metric/group combinations.
Full plotting values, bounds and omission flags are in plot_data.csv.

Main files: frame-error-vs-dimension-scienceplots-log.(png/pdf) and
normalized-frame-error-vs-dimension-scienceplots-log.(png/pdf). Linear companions
use the same names with '-linear'. Rotation-error and normalized-rotation-error
linear figures are also saved using SciencePlots. All values are read from the
saved aggregated_results.csv before plotting; no smoothing or recomputation
from different trials is involved.

## Validation, convergence and old-result comparison

Exactly 240 unique ordered trials and 12 groups of 20; all recorded numeric
fields and aggregate statistics must be finite. Frame/generator checks run
before trials. Full-scan pair-evaluation counts and the one-start fields are
checked per row. Normalized metrics are checked against their raw definitions.
The source file hashes protect the unchanged core; preservation_check.json
compares every previously existing script/result/figure captured before this
run. Future new research plots default to SciencePlots through the shared
research_plot_style context and repository AGENTS.md; old scripts are untouched.

Convergence summary (no PCA/final-MMD gradient failures and no greedy safety
caps in the completed 240-trial run):
```json
{
  "greedy_safety_cap_trials": 0,
  "final_MMD_above_gradient_tolerance": 0,
  "PCA_X_above_gradient_tolerance": 0,
  "PCA_Y_above_gradient_tolerance": 0,
  "final_stopping_criteria": {
    "Terminated - min grad norm reached after 1 iterations": 231,
    "Terminated - min grad norm reached after 3 iterations": 1,
    "Terminated - min grad norm reached after 2 iterations": 8
  },
  "maximum_greedy_accepted_moves": 6,
  "maximum_final_iterations": 3
}
```
Tolerance failures or iteration/safety caps are recorded, never solved by
changing settings or excluding trials. Detailed optimizer stopping conditions
remain in trial_results.csv.

Identical-configuration reference comparison:
```json
{
  "reference_csv": "C:\\Users\\22898\\Desktop\\project\\global-alignment\\numerical_results\\best_improvement_controlled\\trial_results.csv",
  "reference_rows": 61,
  "matching_rows": 20,
  "max_absolute_metric_difference": 0.0,
  "scope": "Only overlapping identical per-trial settings and best-improvement strategy; no comparisons of d=4/8/16, first-improvement, multi-start or runtimes. The old trial CSV may be partial even if an old aggregate CSV exists."
}
```
Old aggregate files are not assumed to belong to a partial/replaced old trial
CSV. Only eligible saved trial rows are compared. New dimensions do not
reproduce different old dimensions; first-improvement and old multi-start
outputs are intentionally not reproduction references.

## Reproduction and files

Python executable: C:\Users\22898\Desktop\project\global-alignment\data\.venv-emdb-baselines\Scripts\python.exe. Software versions, source hashes,
thread environment, exact config and elapsed time are in run_metadata.json.
Only SciencePlots was installed (`pip install --no-deps SciencePlots`); existing
NumPy/SciPy/Matplotlib/Pymanopt/Autograd versions were preserved. Registration
timing uses the original run_trial scope (generation through evaluation);
plotting, checkpoint I/O and imports are outside trial timings.

Run with the validated environment:
```powershell
& .\data\.venv-emdb-baselines\Scripts\python.exe -B numerical_experiments_scienceplots.py
```
Output must be new/empty; --resume validates an existing prefix and skips
completed trials. --plot-only re-renders only this isolated experiment's own
validated figures; no old experimental outputs are touched. The run creates
trial_results.csv, aggregated_results.csv, generator_validation.csv,
plot_data.csv, run_metadata.json, validation.json, convergence_summary.json,
preservation hashes/check, this README, and six figures in both PNG/PDF.
No Overleaf files or manuscript text were edited.
