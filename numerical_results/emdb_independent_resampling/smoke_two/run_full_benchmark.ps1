Set-Location 'C:\Users\22898\Desktop\project\global-alignment'
Remove-Item Env:OPENBLAS_NUM_THREADS, Env:OMP_NUM_THREADS, Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
& '.\data\.venv-emdb-baselines\Scripts\python.exe' -B '.\emdb_independent_resampling_comparison.py' `
  --manifest '.\data\emdb_9515_resampling_pairs\resampling_manifest.csv' `
  --num-pairs 20 `
  --resume-from '.\numerical_results\emdb_independent_resampling\smoke_two\baseline_trial_results.csv' `
  --goicp-convention-record '.\numerical_results\emdb_goicp_feasibility\final_convention_status.json' `
  --output-dir '.\numerical_results\emdb_independent_resampling'
