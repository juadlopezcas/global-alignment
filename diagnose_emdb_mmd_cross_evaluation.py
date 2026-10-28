"""Evaluation-only cross-check of saved EMDB shared-start rotations under MMD.

No registration, optimizers, sampling, candidate modification or bandwidth
tuning. Reuse RotationMMD.evaluate and the original frozen-pair preprocessing.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from importlib import metadata
import inspect
import json
from pathlib import Path
import re
from time import perf_counter
from unittest.mock import patch

import numpy as np

import emdb_baseline_comparison as core
import emdb_independent_resampling_comparison as frozen
import emdb_shared_initialization_comparison as shared
import generate_emdb_resampling_pairs as dataset
from research_plot_style import research_style

ROOT = Path(__file__).resolve().parent
SOURCE = shared.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_mmd_cross_evaluation"
CANDIDATES = ("R_init", "R_mmd", "R_icp", "R_cpd", "R_true")
SOURCE_CONDITIONS = {"R_mmd":"mmd_shared", "R_icp":"icp_shared", "R_cpd":"cpd_shared"}
SIGMA = 1.0
LOSS_TOLERANCE = 1e-12  # Predeclared absolute arithmetic comparison tolerance.
CONSISTENCY_TOLERANCE = 3e-14
ANGLE_TOLERANCE_DEGREES = 1e-8


@contextmanager
def evaluation_only_guard():
    """An accidental call to any existing fit/sampling entry point is an error."""
    forbidden = [(core,name) for name in ('run_ours','register_point_clouds','estimate_pca_frame','greedy_sign_refinement',
                  'run_icp','run_rigid_cpd','run_goicp','random_rotation','validate_goicp_convention')]
    forbidden += [(shared,name) for name in ('shared_initialization','mmd_refinement','icp_refinement','cpd_refinement','synthetic_validation','run')]
    forbidden += [(dataset,name) for name in ('generate_pairs','sample_cloud','random_rotation')]
    forbidden += [(frozen,name) for name in ('run_experiment','iter_pair_results')]
    forbidden.append((core.TrustRegions,'run'))
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(module,name,side_effect=AssertionError(f'Evaluation only: forbidden {name}')))
                  for module,name in forbidden]
        yield
        for guard in guards:guard.assert_not_called()


def classify_alternative(delta):
    """delta=L(alternative)-L(MMD); preserve its raw value separately."""
    if not np.isfinite(delta):raise ValueError('Nonfinite loss difference.')
    return 'MMD_lower' if delta>LOSS_TOLERANCE else 'alternative_lower' if delta < -LOSS_TOLERANCE else 'numerical_tie'


def load_validated_inputs():
    """Verify original/reference/shared provenance and cache preprocessing once/pair."""
    pairs,signature,_ = frozen.preflight(frozen.DEFAULT_MANIFEST,20)
    meta = json.loads((SOURCE/'experiment_metadata.json').read_text())
    prior = meta['signature']
    if meta['status']!='completed' or meta['num_pairs']!=20 or meta['rows']!=140:
        raise ValueError('Expected the completed 20-pair, 140-row shared-start experiment.')
    if prior['reference_signature']!=signature or prior['new_driver_sha256']!=dataset.sha256(Path(shared.__file__)):
        raise ValueError('Source experiment data/settings/code no longer match its signature.')
    if (prior['reference_csv_sha256']!=dataset.sha256(shared.REFERENCE)
            or prior['reference_metadata_sha256']!=dataset.sha256(shared.REFERENCE.parent/'baseline_run_metadata.json')):
        raise ValueError('Original benchmark reference changed.')
    from pycpd import RigidRegistration
    if prior['cpd_library_sha256']!=dataset.sha256(Path(inspect.getfile(RigidRegistration))):
        raise ValueError('Validated CPD implementation changed.')
    inventory = json.loads((SOURCE/'output_manifest.json').read_text())
    inventory_hashes = {row['path']:row['sha256'] for row in inventory['files']}
    for name in ('trial_results.csv','initialization_diagnostics.csv','experiment_metadata.json'):
        path = SOURCE/name
        if inventory_hashes[str(path.resolve())]!=dataset.sha256(path):
            raise ValueError(f'Saved shared-start output hash mismatch: {name}')
    rows = dataset.read_manifest(SOURCE/'trial_results.csv')
    initializations = dataset.read_manifest(SOURCE/'initialization_diagnostics.csv')
    if len(initializations)!=20:raise ValueError('Expected 20 saved common initializations.')
    prepared = {}
    original_prepare = frozen.prepare_pair
    def prepare_once(pair, manifest):
        if Path(manifest).resolve()!=frozen.DEFAULT_MANIFEST.resolve():raise ValueError('Unexpected manifest.')
        if pair['pair_id'] not in prepared:
            prepared[pair['pair_id']] = original_prepare(pair,manifest)
        return prepared[pair['pair_id']]
    with patch.object(frozen,'prepare_pair',side_effect=prepare_once):
        verified = frozen.load_checkpoint(shared.REFERENCE,pairs,frozen.DEFAULT_MANIFEST,20,signature)
        saved = {(r['pair_id'],r['method']):r for r in verified}
        shared.validate_rows(rows,initializations,pairs,saved,20,complete=True)
    if len(prepared)!=20:raise ValueError('Preprocessing cache must contain exactly 20 frozen pairs.')
    return pairs,rows,initializations,prepared,signature


def saved_candidates(pair, initial, rows, prepared):
    """Extract exact matrices from SHARED rows; no composition/cleanup changes."""
    Xc,Yc,truth,scale = prepared
    lookup = {r['method_condition']:r for r in rows if r['pair_id']==pair['pair_id']}
    if len(lookup)!=7:raise ValueError('Duplicate/missing source condition.')
    rotations = {'R_init':np.array(json.loads(initial['R_init']),dtype=np.float64)}
    for candidate,condition in SOURCE_CONDITIONS.items():
        row=lookup[condition]
        if row['condition']!='shared' or row['method_condition']!=condition:
            raise ValueError('Candidate must come from the shared-start condition.')
        if row['R_true']!=pair['R_true'] or row['source_sha256']!=pair['source_sha256'] or row['target_sha256']!=pair['target_sha256']:
            raise ValueError('Candidate belongs to a different frozen pair.')
        np.testing.assert_array_equal(json.loads(row['R_init']),rotations['R_init'])
        if row['Xc_sha256']!=frozen.array_hash(Xc) or row['Yc_sha256']!=frozen.array_hash(Yc):
            raise ValueError('Candidate preprocessing differs.')
        rotations[candidate]=np.array(json.loads(row['R_est']),dtype=np.float64)
    rotations['R_true']=truth
    for R in rotations.values():
        core.assert_rotation(R,atol=1e-10)
        R.setflags(write=False)
    d=json.loads(lookup['cpd_shared']['diagnostics'])
    incremental=np.array(d['cpd_R_incremental'],dtype=np.float64)
    core.assert_rotation(incremental,atol=1e-10)
    np.testing.assert_allclose(rotations['R_cpd'],incremental@rotations['R_init'],atol=1e-12,rtol=0)
    two_stage=(Xc@rotations['R_init'].T)@incremental.T
    one_stage=Xc@rotations['R_cpd'].T
    composition_error=float(np.max(np.abs(two_stage-one_stage)))
    if composition_error>1e-12:raise ValueError('CPD row-convention composition failed.')
    # Scale/translation are checked only for provenance/geometry, never used in loss.
    s=float(lookup['cpd_shared']['scale']);t=np.array(json.loads(lookup['cpd_shared']['translation']))
    np.testing.assert_allclose(s*two_stage+t,s*one_stage+t,atol=1e-12,rtol=0)
    return rotations,lookup,composition_error


def evaluate_candidates(Xc, Yc, rotations, truth):
    """Rotation-only evaluation on one unchanged pair; no t/s input is accepted."""
    before=(frozen.array_hash(Xc),frozen.array_hash(Yc))
    candidate_hashes={k:frozen.array_hash(v) for k,v in rotations.items()}
    objective=core.RotationMMD(Xc,Yc,SIGMA)
    assert objective.X is Xc and objective.Y is Yc
    values=[]
    for name in CANDIDATES:
        R=rotations[name]
        value=objective.evaluate(R)
        repeated=objective.evaluate(R)
        full=core.mmd_squared(Xc@R.T,Yc,SIGMA)
        np.testing.assert_allclose(value,full,atol=CONSISTENCY_TOLERANCE,rtol=0)
        if repeated!=value:raise ValueError('Repeated loss evaluation is not reproducible.')
        angular=core.rotation_angle_error_degrees(truth,R)
        if not np.isfinite([value,full,angular]).all():raise ValueError('Nonfinite evaluation.')
        values.append({'candidate':name,'empirical_mmd':value,'angular_error_degrees':angular,
                       'frobenius_rotation_error':float(np.linalg.norm(truth-R)),
                       'independent_mmd_squared':full,'independent_mmd_absolute_discrepancy':abs(value-full),
                       'repeated_mmd_discrepancy':abs(value-repeated),'R':json.dumps(R.tolist()),
                       'rotation_sha256':candidate_hashes[name]})
    if before!=(frozen.array_hash(Xc),frozen.array_hash(Yc)) or candidate_hashes!={k:frozen.array_hash(v) for k,v in rotations.items()}:
        raise RuntimeError('An input/candidate changed during evaluation.')
    return values


def pair_differences(pair_id,values):
    by_name={r['candidate']:r for r in values}
    loss={k:r['empirical_mmd'] for k,r in by_name.items()}
    angles={k:r['angular_error_degrees'] for k,r in by_name.items()}
    row={'pair_id':pair_id,**{f'L_{name[2:]}':value for name,value in loss.items()},
         **{f'angle_{name[2:]}_degrees':value for name,value in angles.items()}}
    for name in ('icp','cpd','true'):
        delta=loss['R_'+name]-loss['R_mmd']
        row['Delta_'+name]=delta
        row['classification_'+name]=classify_alternative(delta)
        row['MMD_lower_loss_but_higher_angle_than_'+name]=(delta>LOSS_TOLERANCE and angles['R_mmd']>angles['R_'+name]+ANGLE_TOLERANCE_DEGREES)
    row['Delta_mmd_init']=loss['R_mmd']-loss['R_init']
    row['MMD_reduced_loss_vs_init']=row['Delta_mmd_init'] < -LOSS_TOLERANCE
    row['MMD_worsened_angle_vs_init']=angles['R_mmd']>angles['R_init']+ANGLE_TOLERANCE_DEGREES
    for label,names in (('refinements',('R_mmd','R_icp','R_cpd')),('all_candidates',CANDIDATES)):
        minimum=min(loss[name] for name in names)
        row['lowest_'+label+'_within_tolerance']=';'.join(name for name in names if loss[name]<=minimum+LOSS_TOLERANCE)
    return row


def aggregate(pairs):
    summaries=[]
    for name in ('icp','cpd','true','mmd_init'):
        values=np.array([row['Delta_'+name] for row in pairs])
        inverse=name=='mmd_init'
        mmd_lower=values < -LOSS_TOLERANCE if inverse else values > LOSS_TOLERANCE
        alternative_lower=values > LOSS_TOLERANCE if inverse else values < -LOSS_TOLERANCE
        summaries.append({'comparison':'MMD_minus_init' if inverse else name.upper()+'_minus_MMD','num_pairs':len(pairs),
                          'mean_difference':float(values.mean()),'sample_std_difference':float(values.std(ddof=1)),
                          'median_difference':float(np.median(values)),'min_difference':float(values.min()),'max_difference':float(values.max()),
                          'raw_positive_count':int(np.sum(values>0)),'raw_negative_count':int(np.sum(values<0)),
                          'MMD_lower_loss_count':int(mmd_lower.sum()),'alternative_lower_loss_count':int(alternative_lower.sum()),
                          'numerical_tie_count':int(np.sum(np.abs(values)<=LOSS_TOLERANCE)),
                          'MMD_lower_loss_but_higher_angle_count':sum(row['MMD_lower_loss_but_higher_angle_than_'+name] for row in pairs) if not inverse else
                              sum(row['MMD_reduced_loss_vs_init'] and row['MMD_worsened_angle_vs_init'] for row in pairs),
                          'absolute_comparison_tolerance':LOSS_TOLERANCE,
                          'positive_difference_means':'initialization lower loss' if inverse else 'MMD lower loss'})
    return summaries


def lowest_counts(pairs):
    records=[]
    for candidate in ('R_mmd','R_icp','R_cpd'):
        winners=[r['lowest_refinements_within_tolerance'].split(';') for r in pairs]
        records.append({'candidate':candidate,'strict_lowest_among_three_refinements':sum(w==[candidate] for w in winners),
                        'within_tolerance_of_lowest_among_three':sum(candidate in w for w in winners),
                        'strict_lowest_among_all_five':sum(r['lowest_all_candidates_within_tolerance']==candidate for r in pairs),
                        'num_pairs':len(pairs)})
    return records


def plots(candidates,pairs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator,ScalarFormatter
    formats={'R_init':('D','#777777','Init'),'R_mmd':('o','#0C5DA5','MMD'),
             'R_icp':('s','#00B945','ICP'),'R_cpd':('^','#FF9500','CPD'),'R_true':('*','black','Truth (oracle)')}
    with research_style():
        fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
        x=np.arange(len(pairs))
        for name,marker,color in (('icp','s',formats['R_icp'][1]),('cpd','^',formats['R_cpd'][1])):
            ax.plot(x,[1e6*r['Delta_'+name] for r in pairs],marker+'-',color=color,markersize=4,lw=1,label=name.upper()+' minus MMD')
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(x,[r['pair_id'].split('_')[-1] for r in pairs],rotation=45)
        ax.set(xlabel='Frozen pair index',ylabel=r'$(L_{\mathrm{alternative}}-L_{\mathrm{MMD}})\times10^6$',
               title='Rotation-only empirical MMD: positive differences favor MMD')
        ax.legend();ax.grid(alpha=.2)
        for ext in ('png','pdf'):fig.savefig(output/f'mmd-loss-difference-by-pair.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)
        fig,axes=plt.subplots(4,5,figsize=(12,9),sharex=True)
        fig.subplots_adjust(left=.075,right=.985,bottom=.075,top=.90,wspace=.4,hspace=.43)
        handles=[]
        xmax=max(r['angular_error_degrees'] for r in candidates)*1.06
        for index,(ax,pair) in enumerate(zip(axes.flat,pairs)):
            for name,(marker,color,label) in formats.items():
                row=next(r for r in candidates if r['pair_id']==pair['pair_id'] and r['candidate']==name)
                point=ax.scatter(row['angular_error_degrees'],1e6*row['empirical_mmd'],marker=marker,
                                 s=40 if name=='R_true' else 23,edgecolors=color,
                                 facecolors='none' if name in ('R_icp','R_cpd') else color,linewidths=.8,label=label)
                if index==0:handles.append(point)
            ax.set_title(pair['pair_id'],fontsize=9)
            ax.set_xlim(-.35,xmax);ax.set_xticks([0,5,10,15]);ax.tick_params(labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.grid(alpha=.18)
        fig.suptitle('Fixed saved rotations: loss versus angular error within each pair',fontsize=12,y=.995)
        fig.legend(handles,[item[2] for item in formats.values()],loc='upper center',bbox_to_anchor=(.5,.966),ncol=5,fontsize=10)
        fig.supxlabel('Angular rotation error (degrees)',fontsize=11,y=.018)
        fig.supylabel(r'Empirical MMD$^2\times10^6$ (rotation only)',fontsize=11,x=.012)
        for ext in ('png','pdf'):fig.savefig(output/f'mmd-loss-vs-angular-error-by-pair.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
        values=np.array([r['Delta_true'] for r in pairs])
        colors=['#0C5DA5' if v>LOSS_TOLERANCE else '#FF2C00' if v < -LOSS_TOLERANCE else 'gray' for v in values]
        ax.bar(x,1e6*values,color=colors,width=.7)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(x,[r['pair_id'].split('_')[-1] for r in pairs],rotation=45)
        ax.set(xlabel='Frozen pair index',ylabel=r'$(L_{\mathrm{true}}-L_{\mathrm{MMD}})\times10^6$',
               title='Oracle truth under empirical MMD: positive differences favor MMD')
        ax.grid(axis='y',alpha=.2)
        for ext in ('png','pdf'):fig.savefig(output/f'ground-truth-mmd-difference-by-pair.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)


def report(candidates,pairs,summaries,wins,validation,output,elapsed):
    win_table=['| Candidate | Strict lowest loss among MMD/ICP/CPD | Lowest/tied within tolerance |', '|---|---:|---:|']
    for r in wins:win_table.append(f"| {r['candidate']} | {r['strict_lowest_among_three_refinements']}/20 | {r['within_tolerance_of_lowest_among_three']}/20 |")
    diff_table=['| Difference | Mean | Median | Minimum | Maximum | MMD lower / alternative lower / ties |', '|---|---:|---:|---:|---:|---|']
    for r in summaries:
        diff_table.append(f"| {r['comparison']} | {r['mean_difference']:.12e} | {r['median_difference']:.12e} | {r['min_difference']:.12e} | {r['max_difference']:.12e} | "
                          f"{r['MMD_lower_loss_count']} / {r['alternative_lower_loss_count']} / {r['numerical_tie_count']} |")
    detail=['| Candidate on pair_017 | Empirical MMD | Angular error (deg) | MMD difference from R_mmd |','|---|---:|---:|---:|']
    selected=[r for r in candidates if r['pair_id']=='pair_017']
    baseline=next(r['empirical_mmd'] for r in selected if r['candidate']=='R_mmd')
    for r in selected:detail.append(f"| {r['candidate']} | {r['empirical_mmd']:.16e} | {r['angular_error_degrees']:.12f} | {r['empirical_mmd']-baseline:.16e} |")
    by_name={r['comparison']:r for r in summaries}
    optimization_evidence=any(by_name[name]['alternative_lower_loss_count']>0 for name in ('ICP_minus_MMD','CPD_minus_MMD'))
    mismatch_evidence=any(by_name[name]['MMD_lower_loss_but_higher_angle_count']>0 for name in ('ICP_minus_MMD','CPD_minus_MMD'))
    explanation=('Both an observed-candidate optimization limitation and empirical objective mismatch have evidence.' if optimization_evidence and mismatch_evidence
                 else 'The observed comparisons better support empirical MMD objective mismatch; none of the observed ICP/CPD candidates provides evidence of a lower MMD basin.' if mismatch_evidence
                 else 'An observed-candidate optimization limitation has evidence; the candidate comparisons do not establish objective mismatch.' if optimization_evidence
                 else 'These observed comparisons alone do not distinguish the proposed explanations.')
    text=f"""# Rotation-only MMD cross-evaluation: EMD-9515

Evaluation only: 20 existing frozen pairs, 1000 points/cloud, five exact saved
candidate matrices/pair, sigma=1. No registration, TrustRegions, sampling,
candidate refinement, new normalization or hyperparameter selection occurred.
Go-ICP is optional and deliberately omitted; the required five-candidate
comparison isolates the shared-start refinements without an extra near-SO(3)
native-rotation representation issue. No Go-ICP search or new cleanup was used.

## Observed loss preferences

{chr(10).join(win_table)}

{chr(10).join(diff_table)}

Raw differences are unmodified, in original MMD units. ICP/CPD/TRUE minus MMD
positive means MMD has lower loss. MMD minus initialization negative means the
MMD refinement decreases its own loss. Classification uses the predeclared
absolute tolerance {LOSS_TOLERANCE:.1e}: |difference|<=tolerance is a numerical
tie, not a statement of statistical equivalence. All raw signs and exact
values remain in the CSVs, including negative MMD estimates. The tolerance is
an arithmetic comparison convention, not a practical/significance threshold
and was not selected using angular errors. Angular accuracy comparisons use
{ANGLE_TOLERANCE_DEGREES:.1e} degrees to avoid classifying rounding-only changes.

## Answers to A-D

A. MMD decreases its empirical objective relative to initialization in
{by_name['MMD_minus_init']['MMD_lower_loss_count']}/20 pairs; initialization is
lower in {by_name['MMD_minus_init']['alternative_lower_loss_count']}/20; numerical
ties {by_name['MMD_minus_init']['numerical_tie_count']}/20. Mean MMD-minus-init
change is {by_name['MMD_minus_init']['mean_difference']:.16e}.

B. ICP has lower empirical MMD than the MMD result in
{by_name['ICP_minus_MMD']['alternative_lower_loss_count']}/20 pairs; CPD in
{by_name['CPD_minus_MMD']['alternative_lower_loss_count']}/20. Such a lower
observed candidate would be evidence that the saved MMD result is not the
lowest attainable among these candidates, not a proof of any global minimum.

C. MMD has lower loss but HIGHER angular error than ICP in
{by_name['ICP_minus_MMD']['MMD_lower_loss_but_higher_angle_count']}/20 pairs and
than CPD in {by_name['CPD_minus_MMD']['MMD_lower_loss_but_higher_angle_count']}/20.
These counts compare the SAME rotation-only objective on the SAME arrays;
they are not comparisons of ICP RMSE, CPD likelihood or different objectives.

D. MMD is lower than the imposed truth in
{by_name['TRUE_minus_MMD']['MMD_lower_loss_count']}/20 pairs; truth is lower in
{by_name['TRUE_minus_MMD']['alternative_lower_loss_count']}/20, with
{by_name['TRUE_minus_MMD']['numerical_tie_count']}/20 numerical ties. These are
empirical preferences on the original finite samples, not population MMD claims.

## Pair_017 and the earlier landscape diagnosis

{chr(10).join(detail)}

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
{validation['max_independent_mmd_discrepancy']:.16e}. Duplicate evaluations are
bitwise identical. Saved initial/final MMD and saved final angles agree with
the recomputation; maximum saved final-MMD discrepancy is
{validation['max_saved_MMD_final_discrepancy']:.16e}. The independent check
tolerance is {CONSISTENCY_TOLERANCE:.1e}, distinct from the loss-ranking tolerance.

All five matrices/pair pass strict SO(3) checks without any new projection.
ICP uses the saved shared total rotation. CPD uses the saved total rotation,
R_cpd=R_incremental@R_init, where the old adapter already transposed its internal
row rotation. Coordinate equivalence of the two-stage and composed rotations
is checked; maximum residual {validation['max_CPD_composition_residual']:.16e}.
CPD scale and translation are inspected for provenance/composition only and
NEVER applied when evaluating MMD. ICP translation is likewise not applied.
The loss-evaluation function accepts only point clouds and rotations, no
candidate-specific translation, scale or preprocessing parameters.

The smallest absolute alternative-minus-MMD difference is
{validation['minimum_absolute_alternative_difference']:.16e}. Compare this to
the observed arithmetic discrepancies above. Loss gaps are also reported
relative to numerical tolerances without interpreting a small number as a
sampling confidence interval. No point pairs are treated as independent
statistical observations; this report provides descriptive paired summaries.

## Interpretation, limitations and next experiment

{explanation}

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
the CSVs. Elapsed diagnostic time before final reporting/audit: {elapsed:.3f}s.
No Overleaf files were changed.
"""
    (output/'REPORT.md').write_text(text,encoding='utf-8')


def run(output):
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new/empty output directory; no overwrites.')
    began=perf_counter()
    protected=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list((ROOT/'numerical_results').rglob('*'))+list(dataset.DEFAULT_OUTPUT.iterdir())+[dataset.DEFAULT_MAP]
    hashes={str(p.resolve()):dataset.sha256(p) for p in protected if p.is_file() and output.resolve() not in p.resolve().parents
            and p.name not in ('diagnose_emdb_mmd_cross_evaluation.py','test_emdb_mmd_cross_evaluation.py')}
    with evaluation_only_guard():
        pairs,rows,initializations,prepared,signature=load_validated_inputs()
        output.mkdir(parents=True,exist_ok=True)
        frozen.write_json(output/'protected_file_hashes.json',hashes)
        initial={r['pair_id']:r for r in initializations}
        candidate_rows,pair_rows=[],[]
        final_checks,initial_checks,angle_checks,composition_checks=[],[],[],[]
        for pair in pairs:
            pid=pair['pair_id'];X,Y,truth,scale=prepared[pid]
            rotations,lookup,residual=saved_candidates(pair,initial[pid],rows,prepared[pid])
            composition_checks.append(residual)
            values=evaluate_candidates(X,Y,rotations,truth)
            by_name={r['candidate']:r for r in values}
            final_checks.append(abs(by_name['R_mmd']['empirical_mmd']-float(lookup['mmd_shared']['final_mmd'])))
            initial_checks.append(abs(by_name['R_init']['empirical_mmd']-float(initial[pid]['initial_mmd'])))
            if max(final_checks+initial_checks)>CONSISTENCY_TOLERANCE:raise ValueError('Saved MMD losses are not reproduced.')
            for r in values:
                name=r['candidate']
                source_condition=SOURCE_CONDITIONS.get(name,'shared_initialization' if name=='R_init' else 'oracle_truth')
                if name in SOURCE_CONDITIONS:
                    difference=abs(r['angular_error_degrees']-float(lookup[source_condition]['final_angle_error_degrees']))
                    angle_checks.append(difference)
                    if difference>1e-7:raise ValueError('Saved candidate angle differs.')
                r.update(pair_id=pid,source_condition=source_condition,n_source=len(X),n_target=len(Y),sigma=SIGMA,
                         normalization_scale=scale,Xc_sha256=frozen.array_hash(X),Yc_sha256=frozen.array_hash(Y),
                         translation_applied=False,scale_applied=False)
            candidate_rows.extend(values)
            pair_rows.append(pair_differences(pid,values))
            print(f"{pid}: ICP-MMD={pair_rows[-1]['Delta_icp']:+.12e}, CPD-MMD={pair_rows[-1]['Delta_cpd']:+.12e}, "
                  f"true-MMD={pair_rows[-1]['Delta_true']:+.12e}",flush=True)
        frozen.write_checkpoint(output/'candidate_mmd_values.csv',candidate_rows)
        frozen.write_checkpoint(output/'pairwise_mmd_differences.csv',pair_rows)
        summaries=aggregate(pair_rows);wins=lowest_counts(pair_rows)
        frozen.write_checkpoint(output/'aggregated_diagnostics.csv',summaries)
        frozen.write_checkpoint(output/'lowest_loss_counts.csv',wins)
        old=dataset.read_manifest(ROOT/'numerical_results/emdb_9515_mmd_diagnosis_pair_017/candidate_rotations.csv')
        old_lookup={r['candidate']:r for r in old}
        previous={}
        for name,old_name in (('R_init','R_greedy_initial'),('R_mmd','R_final'),('R_true','R_true')):
            new=next(r for r in candidate_rows if r['pair_id']=='pair_017' and r['candidate']==name)
            np.testing.assert_array_equal(json.loads(new['R']),json.loads(old_lookup[old_name]['R']))
            difference=abs(new['empirical_mmd']-float(old_lookup[old_name]['mmd']))
            if difference>CONSISTENCY_TOLERANCE:raise ValueError('pair_017 earlier diagnosis does not match.')
            previous[name]={'loss_absolute_discrepancy':difference,'rotation_fro_difference':0.}
        validation={'passed':True,'num_pairs':20,'candidate_rows':len(candidate_rows),'pair_rows':len(pair_rows),
                    'sigma':SIGMA,'loss_comparison_tolerance':LOSS_TOLERANCE,'independent_validation_tolerance':CONSISTENCY_TOLERANCE,
                    'angular_comparison_tolerance_degrees':ANGLE_TOLERANCE_DEGREES,
                    'all_candidates_from_correct_shared_condition':True,'rotations_unmodified_and_valid_SO3':True,
                    'preprocessing_calls':len(prepared),'same_preprocessed_arrays_for_all_candidates':True,
                    'max_independent_mmd_discrepancy':max(r['independent_mmd_absolute_discrepancy'] for r in candidate_rows),
                    'max_saved_MMD_final_discrepancy':max(final_checks),'max_saved_MMD_init_discrepancy':max(initial_checks),
                    'max_saved_angle_discrepancy_degrees':max(angle_checks),'max_CPD_composition_residual':max(composition_checks),
                    'minimum_absolute_alternative_difference':min(abs(r['Delta_'+name]) for r in pair_rows for name in ('icp','cpd','true')),
                    'repeated_evaluations_bitwise_equal':True,'pair_017_previous_diagnosis':previous,
                    'all_fitting_sampling_and_TrustRegions_run_calls_forbidden':True,'GoICP_included':False,
                    'source_signature':signature,'source_shared_trial_csv_sha256':dataset.sha256(SOURCE/'trial_results.csv'),
                    'source_initialization_csv_sha256':dataset.sha256(SOURCE/'initialization_diagnostics.csv'),
                    'source_code_sha256':dataset.sha256(Path(__file__)),
                    'software_versions':{name:metadata.version(name) for name in ('numpy','scipy','pymanopt','autograd','matplotlib','SciencePlots')}}
        plots(candidate_rows,pair_rows,output)
        elapsed=perf_counter()-began
        report(candidate_rows,pair_rows,summaries,wins,validation,output,elapsed)
        changed=[path for path,value in hashes.items() if dataset.sha256(Path(path))!=value]
        validation.update(previous_files_unchanged=not changed,changed_previous_files=changed,protected_files=len(hashes),elapsed_seconds=perf_counter()-began)
        if changed:raise RuntimeError('A protected source/result changed.')
        frozen.write_json(output/'validation_results.json',validation)
        frozen.write_json(output/'output_manifest.json',{'directory':str(output),'files':[{'path':str(p.resolve()),'sha256':dataset.sha256(p)} for p in sorted(output.iterdir()) if p.is_file()],
                          'inventory_file':str(output/'output_manifest.json')})
    print(f"Completed evaluation only: {len(candidate_rows)} candidates, {len(pair_rows)} pairs; {elapsed:.2f}s. {output}",flush=True)
    return candidate_rows,pair_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir.resolve())


if __name__=='__main__':
    main()
