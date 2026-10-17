# CPD convergence diagnosis: EMD-9515

Source references: [EMRegistration.register](C:/Users/22898/Desktop/project/global-alignment/data/.venv-emdb-baselines/Lib/site-packages/pycpd/emregistration.py:130), [EMRegistration.__init__](C:/Users/22898/Desktop/project/global-alignment/data/.venv-emdb-baselines/Lib/site-packages/pycpd/emregistration.py:81), [RigidRegistration.update_variance](C:/Users/22898/Desktop/project/global-alignment/data/.venv-emdb-baselines/Lib/site-packages/pycpd/rigid_registration.py:106), [RigidRegistration.update_transform](C:/Users/22898/Desktop/project/global-alignment/data/.venv-emdb-baselines/Lib/site-packages/pycpd/rigid_registration.py:60), [RigidRegistration.transform_point_cloud](C:/Users/22898/Desktop/project/global-alignment/data/.venv-emdb-baselines/Lib/site-packages/pycpd/rigid_registration.py:95).

Installed pycpd 2.0.0 source was inspected and copied into installed_pycpd_sources.txt. `EMRegistration.register` iterates while `iteration < max_iterations and diff > tolerance`. The callback argument `error` is `q`, not `diff`. `RigidRegistration.update_variance` sets `diff = abs(q - qprev)`; convergence is an absolute internal-objective difference, not a variance or parameter change. Default tolerance is 0.001 and cap is 100. First q/diff are +infinity before iteration 1. NaN would also end the package loop, so this diagnostic explicitly rejects nonfinite final values.

The objective computed using the previous variance is q = (xPx - 2*s*tr(A R) + s²*YPY)/(2*sigma²) + (D*Np/2)*log(sigma²). Variance is updated afterward. This EM auxiliary objective need not be monotone across iterations because probabilities/variance change. The package floors nonpositive updated variance to tolerance/10. The original adapter interpreted stopping correctly; tolerance was not changed.

Target X=Yc, source Y=Xc; pycpd transforms s*Xc@R_library+t. R_est=R_library.T, so s*Xc@R_est.T+t matches the benchmark. Identity rotation, zero translation, unit scale, automatic initial variance, w=0, and scale estimation remain unchanged. Ground truth enters only post-fit evaluation; cap selection reads only convergence flags. Every cap uses a fresh fit from the same initialization, not a warm start.

Cap-100 reproduction: 20/20 passed; rotation/scale/translation atol 1e-9, accuracy metrics atol 1e-7, exact iteration/stop criterion. Runtime is not expected to reproduce. Maximum rotation discrepancy: 0.

Current Xc/Yc hashes, cloud file hashes, exact R_true, versions, thread environment and original CSV hash are in input_manifest.json. Exact shared preprocessing and RNG functions are reused from the benchmark. Original metadata has no historical cloud hashes; reproduction of all 20 cap-100 results supplies the retrospective input consistency check. Read-only inputs and before/after hashes verify fits do not mutate data.

Fit timers include setup/fitting and lightweight callback snapshots, exclude preprocessing/evaluation/history disk writes. Callback overhead is included, so these diagnostic runtimes are not strictly identical to uninstrumented baseline timings.

## Convergence (cap 1000 tests only failures at 500)

| Cap | Tested | Converged | Cap hit without convergence | Iterations mean / median / range | Runtime mean / median (s) |
|---|---:|---:|---:|---|---|
| 100 | 20 | 0 | 20 | 100.00 / 100.0 / 100.0–100.0 | 6.890 / 6.943 |
| 250 | 20 | 17 | 3 | 212.40 / 222.0 / 152.0–250.0 | 11.703 / 11.691 |
| 500 | 20 | 19 | 1 | 233.30 / 222.0 / 152.0–500.0 | 14.292 / 12.742 |
| 1000 | 1 | 1 | 0 | 526.00 / 526.0 / 526.0–526.0 | 19.637 / 19.637 |

Selected final cap, solely by convergence: **1000**. All trials satisfy unchanged tolerance.

## Ground-truth metrics (interpretation only)

| Cap | Angular mean / median / min / max (deg) | Success <5° / <10° | Scale mean / median / range |
|---|---|---|---|
| 100 | 146.883509 / 143.583920 / 1.561332 / 178.164403 | 5.0% / 5.0% | 0.813016 / 0.810885 / 0.719493–0.961366 |
| 250 | 145.202365 / 137.778669 / 1.583892 / 178.017729 | 5.0% / 5.0% | 0.823661 / 0.825234 / 0.754237–0.961413 |
| 500 | 139.345354 / 137.778669 / 1.583892 / 178.017729 | 10.0% / 10.0% | 0.829256 / 0.825875 / 0.754237–0.961421 |
| 1000 | 1.584814 / 1.584814 / 1.584814 / 1.584814 | 100.0% / 100.0% | 0.961415 / 0.961415 / 0.961415–0.961415 |

Rotation change into cap 250: mean 0.0875506, median 0.0138205, max 0.718679. Full per-trial rotation, scale, translation and internal-error changes are in cpd_cap_comparisons.csv.

Rotation change into cap 500: mean 0.121223, median 0, max 2.41264. Full per-trial rotation, scale, translation and internal-error changes are in cpd_cap_comparisons.csv.

Rotation change into cap 1000: mean 7.44962e-05, median 7.44962e-05, max 7.44962e-05. Full per-trial rotation, scale, translation and internal-error changes are in cpd_cap_comparisons.csv.

Total rotation change from cap 100 to convergence: mean 0.20438673, median 0.013820512, min 0.00058453522, max 2.4979563. Per-trial changes are in cpd_100_to_convergence_changes.csv.

Actual iterations to convergence: mean 234.60, median 222.0, min 152, max 526; per trial [152, 213, 236, 179, 247, 182, 224, 240, 220, 402, 526, 197, 172, 199, 157, 266, 236, 170, 245, 229].

## Old truncated vs converged CPD

| Quantity | Old cap 100 | Selected converged results |
|---|---:|---:|
| rotation_error mean | 2.58130502 | 2.44593884 |
| normalized_rotation_error mean | 1.49031715 | 1.41216345 |
| rotation_angle_error_degrees mean | 146.88350881 | 139.34520994 |
| runtime_seconds mean | 4.62706707 | 14.13388631 |

Large-angle failures remain: 18/20 above 90 degrees. These converged local failures are retained without restart, GT initialization, sign changes or convention changes.

All-20 converged angular error: mean 139.34520994, sample SD 51.34197112, median 137.77866895, range 1.58389178–178.01772910 degrees; success <5 degrees 10.0%, <10 degrees 10.0%.

All-20 converged scale: mean 0.82925537, sample SD 0.07553528, median 0.82587543, range 0.75423709–0.96141490. Scale estimation was retained even for large-angle local failures.

The cap-1000 experiment tests only failures at 500. All other replacement rows reuse their converged cap-500 fits: the installed loop would stop at exactly the same iteration with a 1000 cap. The combined CSV records the selected cap as cpd_max_iterations and the actually tested cap separately as cpd_diagnostic_tested_cap. No additional CPD fits are needed for already converged trials.

## Nonconverged tail inspection

Each row below summarizes last 50 iterations. q is an EM auxiliary objective; change does not by itself certify progress/convergence. No tolerance changes were made.

- Trial 10, cap 500: q change 0.86652361, diff range 0.0056161048–0.11777769, rotation change 0.0036239647, scale change 0.00017191408, translation change 0.00013982426.
  q increased on 44 of the last 49 transitions, so it was not monotonically decreasing. Last five absolute objective differences: [0.008960213033788023, 0.008353359809007088, 0.007784760947288305, 0.007252402096128208, 0.006754315952093748]. Final per-iteration changes: rotation 6.8161028e-06, scale 5.1223232e-07, translation 3.6837675e-07, variance 1.3186849e-08.
  The same identity-initialized trial subsequently met the original tolerance at iteration 526 (diff 0.00099751373). The cap-500 tail therefore represented slow continued convergence, not a persistent numerical stagnation requiring a tolerance change.

Total measured CPD fitting time across all tested caps: 677.329 seconds.

Original benchmark remains untouched. `--combine` validates the reproduction gate, input manifest, selected fits, finite transforms, SO(3), scale/translation and all original non-CPD fields, then writes replacements and combined results into a new directory and calls existing aggregation/plotting only. No Ours, ICP or Go-ICP fit is invoked.

Validated combined results are in `C:\Users\22898\Desktop\project\global-alignment\numerical_results\emdb_baseline_comparison_cpd_converged`. All 60 non-CPD rows retain every original field, and their aggregates are unchanged. Original cloud/result hashes (8 files) and all 42 protected source functions/classes are unchanged. Four CPD-only tests passed. Combined CSV has 80 rows, including 20 converged CPD replacements; aggregate values were independently recomputed and verified. Three comparison plots were regenerated. Accuracy results are ready to use with the retained local failures; CPD runtime includes observational history callbacks and current machine-load effects, as disclosed above.
