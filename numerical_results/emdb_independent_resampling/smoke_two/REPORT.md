# EMD-9515 independent-resampling experiment: implementation and smoke tests

Only pair_000 and pair_001 were fitted. All 20 pairs were generated and frozen first. The full 20-pair benchmark has NOT been run.

## Sampling and frozen protocol

Eligible voxel count: **78351**, threshold **density > 0.035**, map `C:\Users\22898\Desktop\project\global-alignment\data\emd_9515.map`. Scalar voxel_size.x used: **1.3200000524520874**.

[map.py::map_to_point_cloud](C:/Users/22898/Desktop/project/global-alignment/map.py:12) is the authoritative sampling reference. [load_population](C:/Users/22898/Desktop/project/global-alignment/generate_emdb_resampling_pairs.py:47) preserves mrcfile reading, z_idx,y_idx,x_idx=np.where(mrc.data>0.035), column_stack((x_idx,y_idx,z_idx))*mrc.voxel_size.x. No origin/axis corrections, anisotropic voxel-size correction, density weighting, interpolation, Gaussian noise, or perturbations. [sample_cloud](C:/Users/22898/Desktop/project/global-alignment/generate_emdb_resampling_pairs.py:61) uses rng.choice(num_candidates,1000,replace=False). XYZ serialization uses %.4f. Clouds are loaded from these quantized files before rotation/preprocessing.

Master seed=42. [stream_seeds](C:/Users/22898/Desktop/project/global-alignment/generate_emdb_resampling_pairs.py:35) uses SeedSequence([42,pair_index]).spawn(4), then child.generate_state(1,uint64)[0] as the recorded integer default_rng seed. Children 0,1,2,3 respectively control X sampling, Z sampling, artificial rotation, and Ours fit/PCA initialization. All 80 integer seeds are distinct. Method execution/consumption cannot alter samples or R_true. R_true uses the unchanged emdb_baseline_comparison.random_rotation Gaussian-QR function. The actual SO(3) matrix is frozen in the manifest; benchmark loading never regenerates it.

Manifest: [resampling_manifest.csv](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/resampling_manifest.csv). XYZ paths are relative to the manifest so regeneration into a separate directory still gives byte-identical manifest contents. File hashes, map hash, seeds, rotations, and exact source-target overlaps are stored. dataset_metadata.json records the rule and source/version provenance.

## All generated pair files and overlaps

| Pair | Source X | Unrotated target Z | Exact overlap | Source seed | Target seed | Rotation seed |
|---|---|---|---:|---:|---:|---:|
| pair_000 | [pair_000_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_000_X.xyz) | [pair_000_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_000_Z.xyz) | 12 | 16138347438539916964 | 134183728835869882 | 11601846009883706861 |
| pair_001 | [pair_001_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_001_X.xyz) | [pair_001_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_001_Z.xyz) | 11 | 6986019147735528610 | 9866395083222397050 | 1685685575057073228 |
| pair_002 | [pair_002_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_002_X.xyz) | [pair_002_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_002_Z.xyz) | 8 | 9179457708230976085 | 780355794447732201 | 9362106841739433245 |
| pair_003 | [pair_003_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_003_X.xyz) | [pair_003_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_003_Z.xyz) | 12 | 18253498976735827365 | 3394092864088966206 | 3925424493928395603 |
| pair_004 | [pair_004_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_004_X.xyz) | [pair_004_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_004_Z.xyz) | 10 | 7505656699377955574 | 442758459511162919 | 8563677861930954501 |
| pair_005 | [pair_005_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_005_X.xyz) | [pair_005_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_005_Z.xyz) | 16 | 13136572345965894221 | 2668640746680678935 | 10281629845113358797 |
| pair_006 | [pair_006_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_006_X.xyz) | [pair_006_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_006_Z.xyz) | 15 | 10227888288176410859 | 15753730042953236159 | 5221275092895143690 |
| pair_007 | [pair_007_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_007_X.xyz) | [pair_007_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_007_Z.xyz) | 5 | 13134525536295371159 | 4375763501038666248 | 118772512129201406 |
| pair_008 | [pair_008_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_008_X.xyz) | [pair_008_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_008_Z.xyz) | 16 | 3533316147406766313 | 10387802878069681306 | 1321127701133742187 |
| pair_009 | [pair_009_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_009_X.xyz) | [pair_009_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_009_Z.xyz) | 16 | 6707204312292905163 | 1405352748651565557 | 15048277999199899064 |
| pair_010 | [pair_010_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_010_X.xyz) | [pair_010_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_010_Z.xyz) | 8 | 5408620726500015583 | 2166194980583393436 | 9816743974915461207 |
| pair_011 | [pair_011_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_011_X.xyz) | [pair_011_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_011_Z.xyz) | 15 | 3060689983149218498 | 1738518904481790399 | 9499696395910083935 |
| pair_012 | [pair_012_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_012_X.xyz) | [pair_012_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_012_Z.xyz) | 17 | 16122096341080111135 | 4638123434191385128 | 17059270861779047190 |
| pair_013 | [pair_013_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_013_X.xyz) | [pair_013_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_013_Z.xyz) | 12 | 12481672155410529317 | 15211135241469977405 | 12435879012775146016 |
| pair_014 | [pair_014_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_014_X.xyz) | [pair_014_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_014_Z.xyz) | 18 | 11073924585540499637 | 9634210619695715438 | 12048745489633565425 |
| pair_015 | [pair_015_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_015_X.xyz) | [pair_015_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_015_Z.xyz) | 8 | 5755005025686291087 | 11505270843796865904 | 12251304881390215759 |
| pair_016 | [pair_016_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_016_X.xyz) | [pair_016_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_016_Z.xyz) | 12 | 4995782764020468499 | 18357585989116971993 | 14005877246984173404 |
| pair_017 | [pair_017_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_017_X.xyz) | [pair_017_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_017_Z.xyz) | 14 | 9444009686024241478 | 15578235927152897344 | 16395412951127518001 |
| pair_018 | [pair_018_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_018_X.xyz) | [pair_018_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_018_Z.xyz) | 6 | 9537886075155284675 | 17877728696948631474 | 11698800230992666756 |
| pair_019 | [pair_019_X.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_019_X.xyz) | [pair_019_Z.xyz](C:/Users/22898/Desktop/project/global-alignment/data/emdb_9515_resampling_pairs/pair_019_Z.xyz) | 25 | 10021960267423821785 | 12262028238780663164 | 15458520908361380598 |

Overlap range 5–25, mean 12.8. Overlap is permitted across/within pairs; no disjointness was imposed. Every file has 1000 finite distinct points; all 40 file contents and unordered point sets differ.

## Registration and interpretation

[prepare_pair](C:/Users/22898/Desktop/project/global-alignment/emdb_independent_resampling_comparison.py:41) loads X,Z,R_true, computes Y=Z@R_true.T, and calls the unchanged center_and_normalize once. [iter_pair_results](C:/Users/22898/Desktop/project/global-alignment/emdb_independent_resampling_comparison.py:55) shares the same read-only Xc,Yc with all four adapters. Common evaluation uses the same R_true object; fitting adapters never receive it. Data hashes are checked before/after fitting and are recorded in every row. R_est acts as Xc@R_est.T. CPD retains its established R_library.T conversion, fitted isotropic scale and translation. Go-ICP optimalRotation() retains its direct column-rotation convention and unchanged guarded cleanup.

These errors measure recovery of the artificially imposed population-level rotation. X and Z are independent finite resamples; there is no exact pointwise rigid transformation between the observed finite clouds. The plots do not label these metrics as point-cloud alignment quality.

The new entry point imports the validated registration/core module; it does not invoke the old fixed-pair run_trial or run_experiment. All method settings are unchanged: Ours weighted Riemannian PCA → best-improvement greedy search (100 accepted-move safety cap, complete-scan local-optimum stop, tolerance 1e-12) → one TrustRegions MMD solve; sigma=1, final max_iterations=100, PCA max_iterations=200, min_gradient_norm=1e-6. ICP identity, correspondence distance 2.0, Open3D defaults. CPD 2.0.0, cap1000, tolerance0.001, w0, identity/default initialization, scale estimated. Go-ICP DT300, source/model 1000/1000, MSE0.001, trim0, expansion2, translation minimum−0.05 width0.10, rotation cube [−3.1416,3.1416]^3, cleanup guard5e-6. The already-passed Go-ICP synthetic convention record was reused; it was not unnecessarily rerun.

## Smoke results

The first one-pair invocation passed. The second invocation reused its four validated pair_000 rows unchanged and fitted only pair_001. It contains eight rows and two paired-table rows.

| Pair | Method | Imposed-rotation angle (deg) | Frobenius rotation error | Normalized rotation error | Runtime (s) |
|---|---|---:|---:|---:|---:|
| pair_000 | Ours | 4.799839 | 0.11843827 | 0.06838037 | 0.872405 |
| pair_000 | ICP | 131.274287 | 2.57655929 | 1.48757720 | 0.010554 |
| pair_000 | Rigid CPD | 140.719643 | 2.66387486 | 1.53798887 | 16.903349 |
| pair_000 | Go-ICP | 0.371904 | 0.00920109 | 0.00531225 | 103.454715 |
| pair_001 | Ours | 3.856300 | 0.09516586 | 0.05494403 | 0.920259 |
| pair_001 | ICP | 167.958541 | 2.81282556 | 1.62398559 | 0.005073 |
| pair_001 | Rigid CPD | 176.845174 | 2.82735528 | 1.63237433 | 15.475078 |
| pair_001 | Go-ICP | 1.228157 | 0.03030779 | 0.01749821 | 119.053570 |

CPD normal convergence is abs(q_current−q_previous)<=0.001; callback error=q. No tolerance, outlier weight, initialization, scale behavior, or caps were tuned from these errors. Convergence on the remaining 18 pairs has not been measured.

- pair_000 CPD: converged=True, iterations=268, criterion=tolerance, scale=0.7871423750807437, translation norm=0.025318817454047474. Large-angle local failures remain recorded.
- pair_000 Go-ICP: full runtime=103.45471479999833 s (DT=12.93230520002544 s, search=90.50247170007788 s); final native DT-based SSE=2.6048386096954346; translation=[0.000561468756933461, -0.007223655339648621, -0.005666387741938712], norm=0.00919806466997486; nearest translation-cube face margin=0.04277634466035138; SO(3) projection=False. Both margins exceed the fixed 1e-4 boundary-diagnostic threshold comfortably; this threshold is diagnostic only.
- pair_001 CPD: converged=True, iterations=240, criterion=tolerance, scale=0.8511913476021888, translation norm=0.04915888245283193. Large-angle local failures remain recorded.
- pair_001 Go-ICP: full runtime=119.0535701001063 s (DT=11.61382699990645 s, search=107.30593260005116 s); final native DT-based SSE=2.8011996746063232; translation=[-0.0030296548575717555, -0.011430059607010485, -0.01415480202346856], norm=0.01844406385532783; nearest translation-cube face margin=0.03584519797653144; SO(3) projection=False. Both margins exceed the fixed 1e-4 boundary-diagnostic threshold comfortably; this threshold is diagnostic only.

Eight reproducibility/protocol tests passed (including byte-identical repeated generation, distinct streams, reference coordinate conventions, insufficient-population rejection, frozen rotation loading, shared arrays/ground truth, cache rejection, and old-file preservation). 17 original source/cloud/fixed-pair result files retain their exact SHA256 hashes. All returned rotations pass unchanged SO(3) checks; all common trial metrics are finite. The one-pair aggregate sample SD is intentionally undefined; the two-pair aggregates use sample SD. All four plots were generated; the angle-by-pair and Ours/Go-ICP scatter were visually checked. Results from only two pairs do not establish a full independent-pair distribution or justify hyperparameter tuning.

One-pair outputs: `C:\Users\22898\Desktop\project\global-alignment\numerical_results\emdb_independent_resampling\smoke_one`. Two-pair outputs: `C:\Users\22898\Desktop\project\global-alignment\numerical_results\emdb_independent_resampling\smoke_two`. Both contain long-form and aggregate CSVs, pairwise table, run metadata, smoke validation, and all four requested plots.

## Full 20-pair command — provided, NOT executed

This reuses the eight validated smoke rows, then fits only the remaining 18 pairs. The final output will have 80 rows in the requested main directory. Clearing thread variables matches the saved smoke signature. Existing completed output files are never overwritten; changed code/data/settings/versions reject reuse.

```powershell
Set-Location 'C:\Users\22898\Desktop\project\global-alignment'
Remove-Item Env:OPENBLAS_NUM_THREADS, Env:OMP_NUM_THREADS, Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
& '.\data\.venv-emdb-baselines\Scripts\python.exe' -B '.\emdb_independent_resampling_comparison.py' `
  --manifest '.\data\emdb_9515_resampling_pairs\resampling_manifest.csv' `
  --num-pairs 20 `
  --resume-from '.\numerical_results\emdb_independent_resampling\smoke_two\baseline_trial_results.csv' `
  --goicp-convention-record '.\numerical_results\emdb_goicp_feasibility\final_convention_status.json' `
  --output-dir '.\numerical_results\emdb_independent_resampling'
```
