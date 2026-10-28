"""Isolated 240-trial simulated experiment; import the validated best-improvement core.

Only the dimensions and publication plotting change. Existing scripts/results
are protected by hashes. No Overleaf changes, ICP/CPD or perturbation studies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import numerical_experiments as core
from research_plot_style import research_style

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "numerical_results/scienceplots_dimensions_2_4_8_16"
CONFIG = core.ExperimentConfig(
    dimensions=(2, 4, 8, 16), dataset_types=core.DATASET_TYPES,
    n=200, num_trials=20, seed=42, noise_std=.05, sigma=1.,
    max_greedy_moves=100, greedy_tolerance=1e-12, max_iterations=100,
    pca_max_iterations=200, min_gradient_norm=1e-6,
)
EXPERIMENT_ID = "scienceplots_dimensions_2_4_8_16_best_improvement"
PLOT_METRICS = (
    ("frame_error", "Raw frame error (Frobenius norm)", "log"),
    ("normalized_frame_error", r"Normalized frame error ($E_{\mathrm{frame}}/\sqrt{d}$)", "log"),
    ("frame_error", "Raw frame error (Frobenius norm)", "linear"),
    ("normalized_frame_error", r"Normalized frame error ($E_{\mathrm{frame}}/\sqrt{d}$)", "linear"),
    ("rotation_error", "Registration rotation error (Frobenius norm)", "linear"),
    ("normalized_rotation_error", r"Normalized rotation error ($E_{\mathrm{rotation}}/\sqrt{d}$)", "linear"),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def typed_rows(rows):
    strings = {"dataset_type", "greedy_stopping_criterion", "selected_start", "winner_stopping_criterion",
               "pca_X_stopping_criterion", "pca_Y_stopping_criterion"}
    integers = {"dimension", "trial", "seed", "n", "requested_starts", "actual_starts", "max_greedy_moves",
                "max_iterations", "pca_max_iterations", "greedy_num_accepted_moves", "greedy_num_pair_evaluations",
                "winner_iterations", "unconverged_starts"}
    return [{key: value if key in strings else (value is True or value == "True") if key=="greedy_converged"
             else int(value) if key in integers else float(value) for key,value in row.items()} for row in rows]


def validate_trials(rows, complete=True):
    expected = [(kind,d,t) for kind in CONFIG.dataset_types for d in CONFIG.dimensions for t in range(CONFIG.num_trials)]
    keys = [(row["dataset_type"],int(row["dimension"]),int(row["trial"])) for row in rows]
    if keys != expected[:len(rows)] or len(rows)>len(expected) or (complete and len(rows)!=240):
        raise ValueError("Expected the ordered 240-trial design, or its completed prefix.")
    for row in rows:
        for key,value in asdict(CONFIG).items():
            if key not in ("dimensions","dataset_types","num_trials") and float(row[key])!=value:
                raise ValueError(f"Trial setting differs: {key}.")
        if row["selected_start"]!="greedy_best_improvement" or int(row["actual_starts"])!=1 or int(row["requested_starts"])!=1:
            raise ValueError("Trial is not best-improvement with exactly one final MMD start.")
        for key,value in row.items():
            if isinstance(value,(int,float,np.number)) and not np.isfinite(value):
                raise ValueError(f"Nonfinite {key} in {keys}.")
        d=int(row["dimension"])
        np.testing.assert_allclose(row["normalized_frame_error"],row["frame_error"]/np.sqrt(d),atol=1e-15,rtol=0)
        np.testing.assert_allclose(row["normalized_rotation_error"],row["rotation_error"]/np.sqrt(d),atol=1e-15,rtol=0)
        moves=int(row["greedy_num_accepted_moves"])
        scans=moves+int(bool(row["greedy_converged"]))
        if int(row["greedy_num_pair_evaluations"]) != d*(d-1)//2*scans:
            raise ValueError("Greedy proposals do not account for complete BEST-improvement scans.")
    groups=Counter((r["dataset_type"],int(r["dimension"])) for r in rows)
    if complete and (len(groups)!=12 or set(groups.values())!={20}):
        raise ValueError("Expected 12 groups of 20 trials.")
    return groups


def errorbar_plan(means, stds, logarithmic):
    """Never substitute a positive floor for mean-SD; flag omitted lower bars."""
    means,stds=np.asarray(means,dtype=float),np.asarray(stds,dtype=float)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds<0):
        raise ValueError("Mean/SD must be finite; SD must be nonnegative.")
    if logarithmic and np.any(means<=0):
        raise ValueError("A nonpositive mean cannot be represented on a log axis; no clipping allowed.")
    lower,upper=means-stds,means+stds
    return lower,upper,(lower>0) if logarithmic else np.ones(len(means),dtype=bool)


def plot_results(summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, ScalarFormatter, LogLocator, LogFormatterSciNotation
    plot_records=[]
    formats={"gaussian_ellipsoid":("o","-"),"uniform_box":("s","--"),"laplace_product":("^","-.")}
    with research_style():
        assert not plt.rcParams["text.usetex"]
        palette=plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors={kind:palette[i] for i,kind in enumerate(CONFIG.dataset_types)}
        for metric,ylabel,scale in PLOT_METRICS:
            fig,ax=plt.subplots(figsize=(6.8,4.25),layout="constrained")
            omitted=[]
            for kind in CONFIG.dataset_types:
                series=sorted((r for r in summaries if r["dataset_type"]==kind),key=lambda r:r["dimension"])
                x=np.array([r["dimension"] for r in series])
                means=np.array([r[metric+"_mean"] for r in series])
                stds=np.array([r[metric+"_std"] for r in series])
                lower,upper,valid=errorbar_plan(means,stds,scale=="log")
                marker,line=formats[kind]
                curve,=ax.plot(x,means,marker=marker,linestyle=line,markersize=5,
                              linewidth=1.3,color=colors[kind],label=core.DATASET_LABELS[kind])
                color=curve.get_color()
                if valid.any():
                    ax.errorbar(x[valid],means[valid],yerr=stds[valid],fmt="none",capsize=3,color=color,elinewidth=.9)
                for i in np.flatnonzero(~valid):
                    # Only the true mean -> mean+SD interval is drawable. A
                    # hollow down-triangle flags an unrepresentable lower SD.
                    ax.vlines(x[i],means[i],upper[i],color=color,linewidth=.9)
                    ax.plot(x[i],upper[i],marker="_",color=color,markersize=6)
                    ax.plot(x[i],means[i],marker="v",markerfacecolor="none",markeredgecolor=color,markersize=9,linestyle="none")
                    omitted.append((kind,int(x[i])))
                for i,d in enumerate(x):
                    plot_records.append({"figure_metric":metric,"axis_scale":scale,"dataset_type":kind,"dimension":int(d),
                                         "color":color,"marker":marker,"line_style":line,
                                         "num_trials":int(series[i]["num_trials"]),"mean":float(means[i]),"sample_std":float(stds[i]),
                                         "mean_minus_sd":float(lower[i]),"mean_plus_sd":float(upper[i]),
                                         "lower_whisker_drawn":bool(valid[i]),"lower_whisker_policy":"full SD" if valid[i] else "omit nonpositive lower; flag hollow down-triangle; see linear companion"})
            ax.set_yscale(scale)
            ax.xaxis.set_major_locator(FixedLocator(CONFIG.dimensions))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set(xlabel=r"Dimension $d$",ylabel=ylabel+(" (log scale)" if scale=="log" else ""),xlim=(1.5,16.7))
            if scale=="log":
                ax.yaxis.set_major_locator(LogLocator(base=10,numticks=8))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
            else:
                ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,4),useMathText=True)
            ax.grid(axis="y",which="major",alpha=.22,linewidth=.5)
            handles,labels=ax.get_legend_handles_labels()
            if omitted:
                handles.append(Line2D([],[],marker="v",markerfacecolor="none",markeredgecolor="black",linestyle="none"))
                labels.append(r"Mean $-$ SD $\leq0$: lower bar omitted")
            ax.legend(handles,labels,loc="best",fontsize=9,frameon=False)
            ax.set_title("Mean $\\pm$ 1 sample SD; 20 trials per distribution/dimension",fontsize=10)
            stem=f"{metric.replace('_','-')}-vs-dimension-scienceplots-{scale}"
            fig.savefig(output/(stem+".png"),dpi=600)
            fig.savefig(output/(stem+".pdf"))
            plt.close(fig)
    write_csv(output/"plot_data.csv",plot_records)
    return plot_records


def convergence_summary(rows):
    return {"greedy_safety_cap_trials":sum(not r["greedy_converged"] for r in rows),
            "final_MMD_above_gradient_tolerance":sum(r["winner_gradient_norm"]>CONFIG.min_gradient_norm for r in rows),
            "PCA_X_above_gradient_tolerance":sum(r["pca_X_gradient_norm"]>CONFIG.min_gradient_norm for r in rows),
            "PCA_Y_above_gradient_tolerance":sum(r["pca_Y_gradient_norm"]>CONFIG.min_gradient_norm for r in rows),
            "final_stopping_criteria":dict(Counter(r["winner_stopping_criterion"].split(",")[0] for r in rows)),
            "maximum_greedy_accepted_moves":max(r["greedy_num_accepted_moves"] for r in rows),
            "maximum_final_iterations":max(r["winner_iterations"] for r in rows)}


def compare_reference(rows, output):
    """Only same-config BEST-improvement rows at overlapping dimensions; no timings."""
    reference=ROOT/"numerical_results/best_improvement_controlled/trial_results.csv"
    old=typed_rows(read_csv(reference)) if reference.exists() else []
    lookup={(r["dataset_type"],r["dimension"],r["trial"]):r for r in old}
    compared=[]
    fields=("seed","n","noise_std","sigma","max_greedy_moves","greedy_tolerance",
            "max_iterations","pca_max_iterations","min_gradient_norm","selected_start","requested_starts","actual_starts")
    metrics=("frame_error","normalized_frame_error","rotation_error","normalized_rotation_error","final_mmd")
    for row in rows:
        key=(row["dataset_type"],row["dimension"],row["trial"])
        prior=lookup.get(key)
        if prior is None or any(prior.get(k)!=row.get(k) for k in fields):continue
        compared.append({"dataset_type":key[0],"dimension":key[1],"trial":key[2],
                         **{metric+"_new_minus_reference":row[metric]-prior[metric] for metric in metrics}})
    if compared:write_csv(output/"matching_reference_comparison.csv",compared)
    return {"reference_csv":str(reference),"reference_rows":len(old),"matching_rows":len(compared),
            "max_absolute_metric_difference":max((abs(r[m+"_new_minus_reference"]) for r in compared for m in metrics),default=None),
            "scope":"Only overlapping identical per-trial settings and best-improvement strategy; no comparisons of d=4/8/16, first-improvement, multi-start or runtimes. The old trial CSV may be partial even if an old aggregate CSV exists."}


def write_readme(output,rows,summaries,plots,meta,convergence,comparison):
    table=["| Distribution | d | Raw frame error: mean +/- sample SD | Normalized frame error: mean +/- SD |",
           "|---|---:|---:|---:|"]
    for r in summaries:
        table.append(f"| {core.DATASET_LABELS[r['dataset_type']]} | {r['dimension']} | {r['frame_error_mean']:.8g} +/- {r['frame_error_std']:.8g} | "
                     f"{r['normalized_frame_error_mean']:.8g} +/- {r['normalized_frame_error_std']:.8g} |")
    omitted=[r for r in plots if r['axis_scale']=='log' and not r['lower_whisker_drawn']]
    observations=[]
    for kind in CONFIG.dataset_types:
        series=sorted((r for r in summaries if r['dataset_type']==kind),key=lambda r:r['dimension'])
        raw=np.array([r['frame_error_mean'] for r in series])
        norm=np.array([r['normalized_frame_error_mean'] for r in series])
        raw_trend='increases at each tested dimension' if np.all(np.diff(raw)>0) else 'is not monotone across the tested dimensions'
        norm_trend='increases at each tested dimension' if np.all(np.diff(norm)>0) else 'is not monotone across the tested dimensions'
        observations.append(f"- {core.DATASET_LABELS[kind]}: raw mean {raw_trend}, from {raw[0]:.8g} at d=2 to {raw[-1]:.8g} at d=16. Normalized mean {norm_trend} ({norm[0]:.8g} to {norm[-1]:.8g}).")
    per_dimension=[]
    for d in CONFIG.dimensions:
        group=[r for r in summaries if r['dimension']==d]
        best=min(group,key=lambda r:r['frame_error_mean'])
        per_dimension.append(f"d={d}: {core.DATASET_LABELS[best['dataset_type']]}")
    observations.append('- Lowest observed raw mean by dimension: '+ '; '.join(per_dimension)+'. This descriptive ranking is not a significance test.')
    largest=max(rows,key=lambda r:r['frame_error'])
    observations.append(f"- Largest individual frame error is {largest['frame_error']:.9g} ({largest['dataset_type']}, d={largest['dimension']}, zero-based trial={largest['trial']}); its registration rotation error is {largest['rotation_error']:.9g}. Wide frame-error SD bars do not imply the same magnitude of physical rotation error. All trials, including this one, remain in the statistics.")
    observations.append('- Gaussian d=8 has raw SD greater than its raw mean. Its raw and normalized log plots therefore flag the omitted nonpositive lower SD whisker; both linear companions retain the complete interval. The Gaussian normalized mean decreases from d=8 to d=16, so normalized frame error is not universally monotone.')
    text=f"""# SciencePlots simulation: dimensions 2, 4, 8, 16

This is a separate 240-trial experiment. The runner imports the existing
`numerical_experiments.run_trial` and `aggregate_results` without copying or
changing registration, sampling, metrics or convergence settings.

## Configuration and scientific definitions

```json
{json.dumps(asdict(CONFIG),indent=2)}
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

{chr(10).join(table)}

{chr(10).join(observations)}

These compare distributions at fixed n/noise/sigma and each dimension. Raw
Frobenius error sums squared axis errors before taking the square root; the
normalized companion helps separate this dimension scaling from average axis
error. Frame errors describe estimated geometric frames, not final registration
rotation recovery. A monotone or distribution ordering should only be described
where the table supports it; these 20-trial summaries are not significance tests.

## Figures and plotting policy

All six figures use the actual SciencePlots {meta['versions']['SciencePlots']}
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
errors. Omitted log lower whiskers: {len(omitted)} metric/group combinations.
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
{json.dumps(convergence,indent=2)}
```
Tolerance failures or iteration/safety caps are recorded, never solved by
changing settings or excluding trials. Detailed optimizer stopping conditions
remain in trial_results.csv.

Identical-configuration reference comparison:
```json
{json.dumps(comparison,indent=2)}
```
Old aggregate files are not assumed to belong to a partial/replaced old trial
CSV. Only eligible saved trial rows are compared. New dimensions do not
reproduce different old dimensions; first-improvement and old multi-start
outputs are intentionally not reproduction references.

## Reproduction and files

Python executable: {meta['python_executable']}. Software versions, source hashes,
thread environment, exact config and elapsed time are in run_metadata.json.
Only SciencePlots was installed (`pip install --no-deps SciencePlots`); existing
NumPy/SciPy/Matplotlib/Pymanopt/Autograd versions were preserved. Registration
timing uses the original run_trial scope (generation through evaluation);
plotting, checkpoint I/O and imports are outside trial timings.

Run with the validated environment:
```powershell
& .\\data\\.venv-emdb-baselines\\Scripts\\python.exe -B numerical_experiments_scienceplots.py
```
Output must be new/empty; --resume validates an existing prefix and skips
completed trials. --plot-only re-renders only this isolated experiment's own
validated figures; no old experimental outputs are touched. The run creates
trial_results.csv, aggregated_results.csv, generator_validation.csv,
plot_data.csv, run_metadata.json, validation.json, convergence_summary.json,
preservation hashes/check, this README, and six figures in both PNG/PDF.
No Overleaf files or manuscript text were edited.
"""
    (output/'README.md').write_text(text,encoding='utf-8')


def run(output,resume=False,plot_only=False):
    import sys
    CONFIG.validate()
    versions={name:metadata.version(name) for name in ('numpy','scipy','matplotlib','pymanopt','autograd','SciencePlots')}
    signature={'config':json.loads(json.dumps(asdict(CONFIG))),'core_sha256':sha256(ROOT/'numerical_experiments.py'),
               'versions':versions,'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    meta_path=output/'run_metadata.json'
    rows=[]
    if resume or plot_only:
        meta=json.loads(meta_path.read_text())
        if meta.get('experiment_id')!=EXPERIMENT_ID or meta['signature']!=signature:
            raise ValueError('Cannot mix this isolated experiment with other data, code or settings.')
        rows=typed_rows(read_csv(output/'trial_results.csv'))
        validate_trials(rows,complete=plot_only)
        if plot_only:
            meta['latest_plotting_runner_sha256']=sha256(Path(__file__))
    else:
        if output.exists() and any(output.iterdir()):raise ValueError('Output is nonempty; no old results will be overwritten.')
        protected=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list((ROOT/'numerical_results').rglob('*'))
        protected=[p for p in protected if p.is_file() and output.resolve() not in p.resolve().parents
                   and p.name not in ('numerical_experiments_scienceplots.py','research_plot_style.py','test_scienceplots_experiment.py','AGENTS.md')]
        hashes={str(p.resolve()):sha256(p) for p in protected}
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'protected_file_hashes.json',hashes)
        meta={'experiment_id':EXPERIMENT_ID,'signature':signature,'versions':versions,'python_executable':sys.executable,
              'runner_sha256':sha256(Path(__file__)),'style_helper_sha256':sha256(ROOT/'research_plot_style.py'),
              'styles':['science','no-latex'],'status':'running','config':asdict(CONFIG)}
        write_json(meta_path,meta)
    started=perf_counter()
    if not plot_only:
        core.validate_frame_error()
        write_csv(output/'generator_validation.csv',core.validate_generators())
        completed=len(rows)
        for index,(kind,d,trial) in enumerate((k,d,t) for k in CONFIG.dataset_types for d in CONFIG.dimensions for t in range(CONFIG.num_trials)):
            if index<completed:continue
            row=core.run_trial(CONFIG,d,trial,kind)
            rows.append(row);validate_trials(rows,complete=False)
            write_csv(output/'trial_results.csv',rows)
            print(f"{len(rows):3d}/240 {kind} d={d:2d} trial={trial+1:02d}: frame={row['frame_error']:.8f}, "
                  f"rotation={row['rotation_error']:.5f}, gradient={row['winner_gradient_norm']:.3e}, "
                  f"time={row['runtime_seconds']:.2f}s",flush=True)
    # Round-trip the authoritative saved CSV, then aggregate with the original function.
    rows=typed_rows(read_csv(output/'trial_results.csv'));groups=validate_trials(rows)
    summaries=core.aggregate_results(rows)
    if not all(np.isfinite(v) for r in summaries for k,v in r.items() if k!='dataset_type'):
        raise ValueError('Nonfinite aggregate statistic.')
    write_csv(output/'aggregated_results.csv',summaries)
    saved_summaries=[{k:v if k=='dataset_type' else int(v) if k in ('dimension','num_trials') else float(v)
                      for k,v in r.items()} for r in read_csv(output/'aggregated_results.csv')]
    plots=plot_results(saved_summaries,output)
    convergence=convergence_summary(rows);write_json(output/'convergence_summary.json',convergence)
    comparison=compare_reference(rows,output)
    meta.update(status='completed',trial_count=len(rows),group_count=len(groups),
                **({'plotting_invocation_seconds':perf_counter()-started} if plot_only else {'invocation_elapsed_seconds':perf_counter()-started}))
    write_readme(output,rows,summaries,plots,meta,convergence,comparison)
    hashes=json.loads((output/'protected_file_hashes.json').read_text())
    changed=[path for path,digest in hashes.items() if sha256(path)!=digest]
    write_json(output/'preservation_check.json',{'passed':not changed,'protected_files':len(hashes),'changed_files':changed})
    validation={'passed':not changed,'trial_count':len(rows),'groups':[{'dataset_type':k[0],'dimension':k[1],'trials':n} for k,n in groups.items()],
                'all_numeric_fields_finite':True,'all_best_improvement_single_final_MMD_start':True,
                'full_greedy_scan_counts_verified':True,'core_sha256':signature['core_sha256'],
                'plot_source':'saved aggregated_results.csv','style_package_version':versions['SciencePlots'],
                'styles':['science','no-latex'],'latex_required':False,'log_nonpositive_lower_bounds_flagged':sum(not r['lower_whisker_drawn'] for r in plots if r['axis_scale']=='log'),
                'png_dpi':600,'pdf_fonttype':42,'reference_comparison':comparison}
    write_json(output/'validation.json',validation);write_json(meta_path,meta)
    if changed:raise RuntimeError('Previously existing files changed; see preservation_check.json.')
    print(f"Completed {len(rows)} trials, {len(groups)} groups, six SciencePlots PNG/PDF figures. {output}",flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--resume',action='store_true')
    modes.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    run(args.output_dir.resolve(),args.resume,args.plot_only)


if __name__=='__main__':
    main()
