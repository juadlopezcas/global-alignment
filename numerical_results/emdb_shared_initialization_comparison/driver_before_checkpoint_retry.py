"""Controlled EMDB-9515 refinements; reuse validated algorithms and cached baselines.

Only new experiment files are written. The common PCA/greedy rotation is made
once per pair; native identity ICP/CPD and global Go-ICP are verified and reused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
from importlib import metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

import emdb_baseline_comparison as core
import emdb_independent_resampling_comparison as frozen
import generate_emdb_resampling_pairs as dataset
from research_plot_style import research_style

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "numerical_results/emdb_shared_initialization_comparison"
REFERENCE = frozen.DEFAULT_OUTPUT / "baseline_trial_results.csv"
CONFIG = frozen.CONFIG
EXECUTION = ("mmd_shared", "icp_shared", "cpd_shared", "mmd_identity", "icp_identity", "cpd_identity", "goicp_global")
PLOT_ORDER = ("mmd_identity", "icp_identity", "cpd_identity", "mmd_shared", "icp_shared", "cpd_shared", "goicp_global")
LABELS = {"mmd_identity":"MMD\nIdentity", "icp_identity":"ICP\nIdentity", "cpd_identity":"CPD\nIdentity",
          "mmd_shared":"MMD\nShared", "icp_shared":"ICP\nShared", "cpd_shared":"CPD\nShared", "goicp_global":"Go-ICP\nGlobal"}
FITTER_ARGUMENTS = ("Xc", "Yc", "R_init")


def shared_initialization(Xc, Yc, fit_seed):
    """No truth argument, no final MMD solve, and original fit RNG draw order."""
    rng = np.random.default_rng(int(fit_seed))
    started = perf_counter()
    tick = perf_counter()
    pca_X = core.estimate_pca_frame(Xc, rng, max_iterations=CONFIG.pca_max_iterations, min_gradient_norm=CONFIG.min_gradient_norm)
    x_time = perf_counter()-tick
    tick = perf_counter()
    pca_Y = core.estimate_pca_frame(Yc, rng, max_iterations=CONFIG.pca_max_iterations, min_gradient_norm=CONFIG.min_gradient_norm)
    y_time = perf_counter()-tick
    greedy = core.greedy_sign_refinement(Xc,Yc,pca_X.frame,pca_Y.frame,CONFIG.sigma,CONFIG.max_greedy_moves,CONFIG.greedy_tolerance)
    R_init = (pca_Y.frame*greedy.signs) @ pca_X.frame.T
    core.assert_rotation(R_init)
    elapsed = perf_counter()-started
    R_init.setflags(write=False)
    return R_init, {"initialization_runtime_seconds":elapsed, "pca_X_runtime_seconds":x_time,
                    "pca_Y_runtime_seconds":y_time, "greedy_runtime_seconds":greedy.runtime_seconds,
                    "U_X":json.dumps(pca_X.frame.tolist()), "U_Y":json.dumps(pca_Y.frame.tolist()),
                    "greedy_signs":json.dumps(greedy.signs.tolist()), "greedy_history":json.dumps(greedy.history),
                    "greedy_accepted_pairs":json.dumps(greedy.accepted_pairs),
                    "greedy_num_accepted_moves":greedy.num_accepted_moves,"greedy_num_pair_evaluations":greedy.num_pair_evaluations,
                    "greedy_converged":greedy.converged,"greedy_stopping_criterion":greedy.stopping_criterion,
                    "pca_X_gradient_norm":pca_X.gradient_norm,"pca_Y_gradient_norm":pca_Y.gradient_norm,
                    "pca_X_stopping_criterion":pca_X.stopping_criterion,"pca_Y_stopping_criterion":pca_Y.stopping_criterion}


def mmd_refinement(Xc, Yc, R_init):
    """Exactly the final block of core.register_point_clouds, without PCA/greedy."""
    started = perf_counter()
    core.assert_rotation(R_init)
    objective = core.RotationMMD(Xc,Yc,CONFIG.sigma)
    problem = objective.problem(core.SpecialOrthogonalGroup(3))
    initial_mmd = objective.evaluate(R_init)
    optimizer = core.TrustRegions(max_iterations=CONFIG.max_iterations, min_gradient_norm=CONFIG.min_gradient_norm,
                                 use_rand=False,verbosity=0)
    result = optimizer.run(problem,initial_point=R_init)
    core.assert_rotation(result.point)
    final_mmd = objective.evaluate(result.point)
    elapsed = perf_counter()-started
    return core.MethodResult(result.point,elapsed,{"initial_mmd":initial_mmd,"final_mmd":final_mmd,
        "iterations":result.iterations,"gradient_norm":float(result.gradient_norm),
        "stopping_criterion":result.stopping_criterion,"converged":bool(result.gradient_norm<=CONFIG.min_gradient_norm),
        "final_TrustRegions_solves":1})


def icp_refinement(Xc, Yc, R_init):
    transform = np.eye(4); transform[:3,:3] = R_init
    result = core.run_icp(Xc,Yc,max_correspondence_distance=CONFIG.icp_max_correspondence_distance,initial_transform=transform)
    np.testing.assert_array_equal(json.loads(result.diagnostics['icp_initial_transformation']),transform)
    return result


def cpd_refinement(Xc, Yc, R_init):
    """Pre-rotate, use unchanged CPD adapter, then compose R_inc @ R_init.

    pycpd 2.0.0's public R argument rejects valid non-positive-definite rotations.
    Its returned R_library is a ROW rotation; the unchanged adapter transposes it
    to R_inc. CPD still estimates its scale/translation with standard s=1,t=0.
    """
    started = perf_counter()
    X_pre = Xc @ R_init.T
    X_pre.setflags(write=False)
    incremental = core.run_rigid_cpd(X_pre,Yc)
    total = incremental.R_est @ R_init
    elapsed = perf_counter()-started
    core.assert_rotation(total)
    s = float(incremental.diagnostics['cpd_scale'])
    t = np.array(json.loads(incremental.diagnostics['cpd_translation']))
    two_stage = s*(X_pre @ incremental.R_est.T)+t
    composed = s*(Xc @ total.T)+t
    residual = float(np.max(np.abs(two_stage-composed)))
    np.testing.assert_allclose(two_stage,composed,atol=1e-10,rtol=1e-10)
    diagnostics = {**incremental.diagnostics,"cpd_initialization_route":"pre-rotation + unchanged default incremental CPD",
                   "cpd_R_incremental":incremental.R_est.tolist(),"cpd_effective_initial_R":R_init.tolist(),
                   "cpd_composition_max_abs_residual":residual,"cpd_incremental_adapter_runtime_seconds":incremental.runtime_seconds}
    return core.MethodResult(total,elapsed,diagnostics)


def cached_result(saved, method):
    """Retain verified saved rotation, original runtime and full native diagnostics."""
    R = np.array(json.loads(saved['R_est']),dtype=float)
    diagnostics = {k:v for k,v in saved.items() if k.startswith(method+'_') and v not in ('',None)}
    if method=='icp':
        for key in ('icp_max_correspondence_distance','icp_max_iterations','icp_relative_fitness','icp_relative_rmse'):
            diagnostics[key] = float(diagnostics[key])
    if method=='cpd':
        for key in ('cpd_scale','cpd_iterations','cpd_max_iterations','cpd_tolerance','cpd_w'):
            diagnostics[key] = float(diagnostics[key])
    return core.MethodResult(R,float(saved['runtime_seconds']),diagnostics)


def verify_mmd_reproduction(result, saved):
    fro = float(np.linalg.norm(result.R_est-np.array(json.loads(saved['R_est']))))
    if fro > 1e-9 or abs(result.diagnostics['final_mmd']-float(saved['final_mmd']))>1e-12:
        raise RuntimeError(f'Shared MMD did not reproduce saved full Ours pipeline (rotation difference {fro}). STOP.')
    if result.diagnostics['iterations'] != int(saved['winner_iterations']):
        raise RuntimeError('Reproduced MMD iteration count differs from the original.')
    return fro


def synthetic_validation():
    """Predetermined geometric checks; synthetic truth never enters any fitter."""
    from pycpd import RigidRegistration
    rng = np.random.default_rng(2026102701)
    X = rng.uniform([-1.,-.5,-.2],[1.,.8,.7],size=(80,3))
    X[:,1] += .25*X[:,0]**2
    truth = Rotation.from_rotvec([.17,-.09,.13]).as_matrix()
    initial = Rotation.from_rotvec([.14,-.06,.12]).as_matrix()  # separately predeclared, not computed from truth.
    Xc,Yc,_ = core.center_and_normalize(X,X @ truth.T)
    rejected = False
    try:
        RigidRegistration(X=Yc,Y=Xc,R=np.diag([-1.,-1.,1.]),max_iterations=1000)
    except ValueError:
        rejected = True
    if not rejected:
        raise RuntimeError('Installed CPD interface differs from inspected 2.0.0 behavior; inspect before fitting.')
    results = {}
    for method,fit in (('icp',icp_refinement),('cpd',cpd_refinement)):
        result = fit(Xc,Yc,initial)
        metrics = core.evaluate_rotation(truth,result.R_est,result.runtime_seconds)
        if metrics['rotation_error']>1e-5 or metrics['rotation_angle_error_degrees']>.01:
            raise RuntimeError(f'{method} synthetic controlled convention failed: {metrics}')
        results[method] = {**metrics,'diagnostics':result.diagnostics}
    return {'passed':True,'seed':2026102701,'points':80,'R_true':truth.tolist(),'R_init':initial.tolist(),
            'cpd_public_R_rejects_valid_SO3':rejected,'cpd_public_signature':str(inspect.signature(RigidRegistration)),
            'public_R_guard_source':inspect.getsource(__import__('pycpd.rigid_registration',fromlist=['is_positive_semi_definite']).is_positive_semi_definite),
            'results':results,'synthetic_initial_rotation_predeclared_independently':True}


def make_row(pair, Xc, Yc, truth, scale, key, result, R_init, init_seconds, reused=False, reproduction=None):
    """Evaluation boundary: truth enters only here, after fitting is finished."""
    method,condition = key.split('_')
    global_search = condition=='global'
    metrics = core.evaluate_rotation(truth,result.R_est,result.runtime_seconds)
    initial_angle = None if global_search else core.rotation_angle_error_degrees(truth,R_init)
    d = dict(result.diagnostics)
    t = np.zeros(3); s = 1.
    if method=='icp':
        t = np.array(json.loads(d['icp_transformation']))[:3,3]
        converged = None
        stopping = 'Open3D 0.19 exposes no convergence flag or actual iteration count; unchanged 30-iteration/default criteria'
    elif method=='cpd':
        t = np.array(json.loads(d['cpd_translation'])); s=float(d['cpd_scale'])
        converged = d['cpd_stopping_criterion']=='tolerance'
        stopping = d['cpd_stopping_criterion']
        if not converged and int(d['cpd_iterations'])!=1000:
            raise RuntimeError('CPD stopped early without tolerance convergence; investigate numerical failure.')
    elif method=='goicp':
        t = np.array(json.loads(d['goicp_translation']))
        converged = None; stopping='verified saved bounded global-search completion; native MSE threshold .001'
    else:
        converged=d['converged']; stopping=d['stopping_criterion']
    if t.shape!=(3,) or not np.isfinite(t).all() or not np.isfinite(s):
        raise RuntimeError('Nonfinite transformation parameters.')
    return {'emdb_id':pair['emdb_id'],'pair_id':pair['pair_id'],'method_condition':key,'method':method,'condition':condition,
            'n_source':len(Xc),'n_target':len(Yc),'normalization_scale':scale,'sigma':CONFIG.sigma,
            'ours_fit_seed':int(pair['ours_fit_seed']),'source_sha256':pair['source_sha256'],'target_sha256':pair['target_sha256'],
            'Xc_sha256':frozen.array_hash(Xc),'Yc_sha256':frozen.array_hash(Yc),
            'R_true':pair['R_true'],'R_init':None if global_search else json.dumps(R_init.tolist()),'R_est':json.dumps(result.R_est.tolist()),
            'initial_angle_error_degrees':initial_angle,'final_angle_error_degrees':metrics['rotation_angle_error_degrees'],
            'rotation_error':metrics['rotation_error'],'normalized_rotation_error':metrics['normalized_rotation_error'],
            'angular_improvement_degrees':None if global_search else initial_angle-metrics['rotation_angle_error_degrees'],
            'registration_runtime_seconds':result.runtime_seconds,
            'refinement_runtime_seconds':None if global_search else result.runtime_seconds,
            'initialization_runtime_seconds':init_seconds if condition=='shared' else None if global_search else 0.,
            'total_pipeline_runtime_seconds':result.runtime_seconds+(init_seconds if condition=='shared' else 0.),
            'execution_runtime_seconds_this_run':0. if reused else result.runtime_seconds,
            'result_reused':reused,'runtime_provenance':'verified prior benchmark measurement' if reused else 'measured in this experiment',
            'translation':json.dumps(t.tolist()),'translation_norm':float(np.linalg.norm(t)),'scale':s,
            'solver_converged':converged,'stopping_criterion':stopping,
            'initial_mmd':d.get('initial_mmd'),'final_mmd':d.get('final_mmd'),
            'reference_R_fro_difference':reproduction,'diagnostics':json.dumps(d)}


def validate_rows(rows, initializations, pairs, saved, num_pairs, complete=False):
    expected=[(p['pair_id'],key) for p in pairs[:num_pairs] for key in EXECUTION]
    if [(r['pair_id'],r['method_condition']) for r in rows]!=expected[:len(rows)] or len(rows)>len(expected):
        raise ValueError('Checkpoint must be an ordered pair/condition prefix, without duplicates.')
    if complete and len(rows)!=num_pairs*7:raise ValueError('Incomplete controlled experiment.')
    init_lookup={r['pair_id']:r for r in initializations}
    expected_init=[p['pair_id'] for p in pairs[:len(initializations)]]
    if list(init_lookup)!=expected_init or len(init_lookup)!=len(initializations):raise ValueError('Initialization checkpoint is not a unique pair prefix.')
    prepared={}
    for init in initializations:
        p=next(p for p in pairs if p['pair_id']==init['pair_id'])
        X,Y,truth,scale=frozen.prepare_pair(p,frozen.DEFAULT_MANIFEST)
        prepared[p['pair_id']]=(X,Y,truth,scale)
        R=np.array(json.loads(init['R_init']))
        Ux,Uy=np.array(json.loads(init['U_X'])),np.array(json.loads(init['U_Y']))
        signs=np.array(json.loads(init['greedy_signs']))
        for rotation in (R,Ux,Uy):core.assert_rotation(rotation)
        if signs.shape!=(3,) or not np.isin(signs,[-1,1]).all() or np.prod(signs)!=1:
            raise ValueError('Initialization signs violate the original proper-rotation convention.')
        np.testing.assert_allclose(R,(Uy*signs)@Ux.T,atol=1e-14,rtol=0)
        if (init['Xc_sha256']!=frozen.array_hash(X) or init['Yc_sha256']!=frozen.array_hash(Y)
                or int(init['ours_fit_seed'])!=int(p['ours_fit_seed'])):
            raise ValueError('Initialization source/seed mismatch.')
        np.testing.assert_allclose(float(init['normalization_scale']),scale,atol=1e-12,rtol=0)
        np.testing.assert_allclose(float(init['initial_angle_error_degrees']),core.rotation_angle_error_degrees(truth,R),atol=1e-10,rtol=0)
    for row in rows:
        p=next(p for p in pairs if p['pair_id']==row['pair_id'])
        if p['pair_id'] not in prepared:prepared[p['pair_id']]=frozen.prepare_pair(p,frozen.DEFAULT_MANIFEST)
        X,Y,truth,scale=prepared[p['pair_id']]
        for key in ('source_sha256','target_sha256','R_true'):
            if row[key]!=p[key]:raise ValueError('Frozen-pair checkpoint mismatch.')
        if row['Xc_sha256']!=frozen.array_hash(X) or row['Yc_sha256']!=frozen.array_hash(Y):raise ValueError('Input hashes differ.')
        final=np.array(json.loads(row['R_est']))
        metric=core.evaluate_rotation(truth,final,float(row['registration_runtime_seconds']))
        for key,value in (('final_angle_error_degrees',metric['rotation_angle_error_degrees']),('rotation_error',metric['rotation_error'])):
            np.testing.assert_allclose(float(row[key]),value,atol=1e-10,rtol=0)
        if row['condition']!='global':
            expected_R=np.array(json.loads(init_lookup[p['pair_id']]['R_init'])) if row['condition']=='shared' else np.eye(3)
            np.testing.assert_array_equal(json.loads(row['R_init']),expected_R)
            initial_angle=core.rotation_angle_error_degrees(truth,expected_R)
            np.testing.assert_allclose(float(row['initial_angle_error_degrees']),initial_angle,atol=1e-10,rtol=0)
            np.testing.assert_allclose(float(row['angular_improvement_degrees']),initial_angle-metric['rotation_angle_error_degrees'],atol=1e-10,rtol=0)
        if row['method_condition']=='mmd_shared':
            np.testing.assert_allclose(final,json.loads(saved[p['pair_id'],'ours']['R_est']),atol=1e-9,rtol=0)
        if row['method_condition'] in ('icp_identity','cpd_identity','goicp_global'):
            prior=saved[p['pair_id'],row['method']]
            np.testing.assert_array_equal(final,json.loads(prior['R_est']))
            assert float(row['registration_runtime_seconds'])==float(prior['runtime_seconds'])
        d=json.loads(row['diagnostics'])
        if row['method_condition']=='icp_shared':
            initial_T=np.array(json.loads(d['icp_initial_transformation']))
            np.testing.assert_array_equal(initial_T[:3,:3],json.loads(row['R_init']))
            np.testing.assert_array_equal(initial_T[:3,3],np.zeros(3))
        if row['method_condition']=='cpd_shared':
            start=np.array(json.loads(row['R_init']))
            np.testing.assert_array_equal(d['cpd_effective_initial_R'],start)
            np.testing.assert_allclose(final,np.array(d['cpd_R_incremental'])@start,atol=1e-12,rtol=0)
            if d['cpd_composition_max_abs_residual']>1e-10:raise ValueError('Saved CPD composition mismatch.')
        expected_total=float(row['registration_runtime_seconds'])+(float(row['initialization_runtime_seconds']) if row['condition']=='shared' else 0.)
        np.testing.assert_allclose(float(row['total_pipeline_runtime_seconds']),expected_total,atol=1e-12,rtol=0)
        for key in ('registration_runtime_seconds','total_pipeline_runtime_seconds','rotation_error','final_angle_error_degrees','scale','translation_norm'):
            if not np.isfinite(float(row[key])):raise ValueError(f'Nonfinite saved {key}.')


def numeric_rows(rows):
    numeric=('normalization_scale','sigma','initial_angle_error_degrees','final_angle_error_degrees','rotation_error',
             'normalized_rotation_error','angular_improvement_degrees','registration_runtime_seconds','refinement_runtime_seconds',
             'initialization_runtime_seconds','total_pipeline_runtime_seconds','execution_runtime_seconds_this_run','translation_norm','scale',
             'initial_mmd','final_mmd','reference_R_fro_difference')
    return [dict(r,**{key:None if r[key] in ('',None) else float(r[key]) for key in numeric}) for r in rows]


def statistics(rows):
    results=[]
    for key in PLOT_ORDER:
        group=[r for r in rows if r['method_condition']==key]
        record={'method_condition':key,'num_pairs':len(group)}
        for field in ('initial_angle_error_degrees','final_angle_error_degrees','rotation_error','angular_improvement_degrees',
                      'registration_runtime_seconds','refinement_runtime_seconds','initialization_runtime_seconds','total_pipeline_runtime_seconds','scale','translation_norm'):
            values=np.array([r[field] for r in group if r[field] is not None])
            for name,value in (('mean',values.mean() if len(values) else None),('std',values.std(ddof=1) if len(values)>1 else None),
                               ('median',np.median(values) if len(values) else None),('min',values.min() if len(values) else None),('max',values.max() if len(values) else None)):
                record[field+'_'+name]=None if value is None else float(value)
        errors=np.array([r['final_angle_error_degrees'] for r in group])
        for limit in (1,5,10):record[f'success_rate_angle_lt_{limit}_deg']=float(np.mean(errors<limit))
        record['solver_converged_count']=sum(r['solver_converged'] is True or r['solver_converged']=='True' for r in group)
        record['solver_convergence_not_exposed']=key.startswith('icp') or key=='goicp_global'
        results.append(record)
    return results


def pairwise(rows,initializations):
    lookup={(r['pair_id'],r['method_condition']):r for r in rows}
    wide=[]
    for init in initializations:
        p=init['pair_id']
        r={'pair_id':p,'shared_initial_angle_degrees':float(init['initial_angle_error_degrees'])}
        for key in PLOT_ORDER:
            r[key+'_angle_degrees']=lookup[p,key]['final_angle_error_degrees']
        wide.append(r)
    comparisons=[]
    pairs=[('mmd_shared','mmd_identity'),('icp_shared','icp_identity'),('cpd_shared','cpd_identity'),
           ('icp_shared','mmd_shared'),('cpd_shared','mmd_shared'),('icp_shared','cpd_shared')]
    pairs += [(key,'shared_initial') for key in ('mmd_shared','icp_shared','cpd_shared')]
    pairs += [(key,'goicp_global') for key in ('mmd_shared','icp_shared','cpd_shared')]
    for a,b in pairs:
        deltas=np.array([r[a+'_angle_degrees']-(r['shared_initial_angle_degrees'] if b=='shared_initial' else r[b+'_angle_degrees']) for r in wide])
        comparisons.append({'first':a,'second':b,'num_pairs':len(wide),'mean_first_minus_second_degrees':float(deltas.mean()),
                            'std_paired_difference_degrees':float(deltas.std(ddof=1)) if len(deltas)>1 else None,
                            'median_paired_difference_degrees':float(np.median(deltas)),
                            'first_lower_count':int(np.sum(deltas < -1e-8)),'second_lower_count':int(np.sum(deltas > 1e-8)),
                            'tie_count_tolerance_1e_8_degrees':int(np.sum(np.abs(deltas)<=1e-8))})
    return wide,comparisons


def plots(rows,wide,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with research_style():
        palette=plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors=[palette[0],palette[1],palette[2],palette[0],palette[1],palette[2],palette[3]]
        fig,ax=plt.subplots(figsize=(8.2,4.6),layout='constrained')
        boxes=ax.boxplot([[r['final_angle_error_degrees'] for r in rows if r['method_condition']==k] for k in PLOT_ORDER],patch_artist=True,medianprops={'color':'black'})
        for i,(box,color) in enumerate(zip(boxes['boxes'],colors)):
            box.set_facecolor(color);box.set_alpha(.55)
            if 3<=i<6:box.set_hatch('//')
        ax.axvline(3.5,color='gray',lw=.6,ls=':');ax.axvline(6.5,color='gray',lw=.6,ls=':')
        ax.set_xticks(range(1,8),[LABELS[k] for k in PLOT_ORDER])
        ax.set(ylabel='Final rotation error (degrees)',title=f'EMD-9515: {len(wide)} frozen pairs; initialization controls')
        ax.grid(axis='y',alpha=.2)
        for ext in ('png','pdf'):fig.savefig(output/f'angular-error-initialization-comparison.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(8,6.2),layout='constrained',sharex=True)
        x=np.arange(len(wide))
        axes[0].plot(x,[r['shared_initial_angle_degrees'] for r in wide],'k--',lw=1,label='Shared PCA + greedy start')
        for key,color,marker in zip(('mmd_shared','icp_shared','cpd_shared'),palette,('o','s','^')):
            final=np.array([r[key+'_angle_degrees'] for r in wide])
            initial=np.array([r['shared_initial_angle_degrees'] for r in wide])
            axes[0].plot(x,final,marker+'-',color=color,markersize=3,lw=.9,label=key.split('_')[0].upper())
            axes[1].plot(x,initial-final,marker+'-',color=color,markersize=3,lw=.9)
        axes[1].axhline(0,color='black',lw=.7)
        axes[0].set(ylabel='Rotation error (degrees)',title='Same starting rotation; all shared-start refinements')
        axes[1].set(xlabel='Frozen pair index',ylabel='Initial minus final error (degrees)\nPositive = improvement')
        axes[1].set_xticks(x,[r['pair_id'].split('_')[-1] for r in wide],rotation=45)
        axes[0].legend(ncol=2,fontsize=9)
        for ax in axes:ax.grid(alpha=.2)
        for ext in ('png','pdf'):fig.savefig(output/f'shared-initialization-refinement-by-pair.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8.2,4.8),layout='constrained')
        for i,(key,color) in enumerate(zip(PLOT_ORDER,colors),1):
            group=[r for r in rows if r['method_condition']==key]
            total=np.array([r['total_pipeline_runtime_seconds'] for r in group])
            registration_times=np.array([r['registration_runtime_seconds'] for r in group])
            assert np.all(total>0) and np.all(registration_times>0)
            box=ax.boxplot([total],positions=[i],widths=.5,patch_artist=True,medianprops={'color':'black'})
            box['boxes'][0].set_facecolor(color);box['boxes'][0].set_alpha(.5)
            if key.endswith('_shared'):
                for a,b in zip(registration_times,total):ax.plot([i-.12,i+.12],[a,b],color=color,alpha=.35,lw=.6)
                ax.scatter(np.full(len(group),i-.12),registration_times,s=12,facecolors='none',edgecolors=color,
                           label='Refinement only (open circles)' if key=='mmd_shared' else None)
        ax.set_xticks(range(1,8),[LABELS[k] for k in PLOT_ORDER])
        ax.set_yscale('log')
        ax.set(ylabel='Runtime (seconds, log scale)',title='Boxes: standalone pipeline runtime; shared overhead included')
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
        for ext in ('png','pdf'):fig.savefig(output/f'runtime-initialization-comparison.{ext}',**({'dpi':600} if ext=='png' else {}))
        plt.close(fig)


def write_report(rows,initializations,summaries,comparisons,ledger,output):
    table=['| Condition | Mean +/- SD angle (deg) | Median (deg) | <1 / <5 / <10 deg | Mean refinement or global runtime (s) | Mean total pipeline (s) |',
           '|---|---:|---:|---|---:|---:|']
    for r in summaries:
        sd='undefined' if r['final_angle_error_degrees_std'] is None else f"{r['final_angle_error_degrees_std']:.6f}"
        table.append(f"| {r['method_condition']} | {r['final_angle_error_degrees_mean']:.6f} +/- {sd} | {r['final_angle_error_degrees_median']:.6f} | "
                     f"{r['success_rate_angle_lt_1_deg']:.2f} / {r['success_rate_angle_lt_5_deg']:.2f} / {r['success_rate_angle_lt_10_deg']:.2f} | "
                     f"{r['registration_runtime_seconds_mean']:.4f} | {r['total_pipeline_runtime_seconds_mean']:.4f} |")
    paired=['| First - second | Mean paired angle difference (deg) | First lower / second lower / ties |','|---|---:|---|']
    for r in comparisons:
        paired.append(f"| {r['first']} - {r['second']} | {r['mean_first_minus_second_degrees']:.6f} | {r['first_lower_count']} / {r['second_lower_count']} / {r['tie_count_tolerance_1e_8_degrees']} |")
    worsening=[r['pair_id'] for r in rows if r['method_condition']=='mmd_shared' and r['angular_improvement_degrees']< -1e-8 and r['final_mmd']<r['initial_mmd']]
    cpd=[r for r in rows if r['method']=='cpd']
    init_mean=float(np.mean([float(r['initial_angle_error_degrees']) for r in initializations]))
    summary={r['method_condition']:r for r in summaries}
    text=f"""# Controlled initialization comparison: EMD-9515

{len(initializations)} consecutive frozen independent-resampling pairs; n=1000.
No map samples or imposed rotations were regenerated. X/Z quantization and
Y=Z@R_true.T, separate centering and shared maximum-radius normalization are
the original functions. All new refinements share the same read-only Xc/Yc
values per pair; CPD pre-rotation only represents its supplied initial rotation.
Ground truth is used only in result evaluation, never passed to a fitter.

## Accuracy and computational cost

{chr(10).join(table)}

Mean common-initialization angle: {init_mean:.9f} degrees. Identity-to-shared
mean angle reductions: MMD {summary['mmd_identity']['final_angle_error_degrees_mean']-summary['mmd_shared']['final_angle_error_degrees_mean']:.9f},
ICP {summary['icp_identity']['final_angle_error_degrees_mean']-summary['icp_shared']['final_angle_error_degrees_mean']:.9f},
CPD {summary['cpd_identity']['final_angle_error_degrees_mean']-summary['cpd_shared']['final_angle_error_degrees_mean']:.9f} degrees.
These within-method contrasts quantify sensitivity to initialization on these
pairs; they do not decompose all between-method differences into causal shares.

{chr(10).join(paired)}

Negative paired first-minus-second difference favors the first method. Sample
SDs, medians and per-pair differences are provided; no significance claims or
general-superiority claims follow from these 20 pairs. Raw rotation error is
||R_true-R_est||_F. Angular errors use the unchanged clipped trace/arccos metric
without signed-permutation equivalence.

MMD reduces empirical rotation-only MMD but worsens angular error in
{len(worsening)} shared-start pairs: {worsening}. Initial and final MMD are
recorded for MMD only; native ICP/CPD/Go-ICP objectives are not comparative
accuracy metrics. A lower empirical loss need not improve imposed-rotation
recovery on independent finite clouds. CPD converged by its native tolerance
in {sum(r['solver_converged'] for r in cpd)}/{len(cpd)} identity/shared rows.
Open3D 0.19's result does not expose actual iteration count or a convergence
flag; those are explicitly unavailable, not fabricated from benchmark accuracy.

## Exact initialization and validated algorithms

The original weighted SO(3) PCA functions are called X then Y with the stored
ours_fit_seed RNG. Original BEST-improvement pairwise greedy examines all
three determinant-preserving flips per scan. R_init=(U_Y*greedy_signs)@U_X.T is
computed once and cached per pair, before any final refinement. The same
matrix starts MMD, ICP and CPD. Sigma=1; PCA max=200; final MMD max=100;
min gradient=1e-6; greedy cap=100 and improvement tolerance=1e-12. The MMD
refinement is exactly one native TrustRegions solve; no extra starts are used.
Its output reproduces the saved full Ours pipeline on every completed pair,
without separately rerunning the full adapter and recomputing initialization.

ICP calls the existing adapter with T_init[:3,:3]=R_init and zero translation,
correspondence distance 2, unchanged point-to-point estimation and Open3D
default convergence criteria (30 iterations, relative fitness/RMSE 1e-6).
It estimates translation, with scale fixed to one as before.

Installed pycpd 2.0.0's public R argument validates positive definiteness and
rejects some valid SO(3) matrices. Thus the documented fallback is used, without
patching the library: X_pre=Xc@R_init.T; the unchanged run_rigid_cpd adapter
fits X_pre to Yc at its default incremental identity, scale=1, translation=0.
Its R_inc is already R_library.T. Compose R_total=R_inc@R_init. For every fit,
s*(X_pre@R_inc.T)+t equals s*(Xc@R_total.T)+t numerically. CPD retains scale and
translation optimization, max_iterations=1000, tolerance=.001, w=0 and default
automatic initial variance. No diagnostic-history callback is added to timing.
Synthetic noncommuting-rotation tests verify the geometry independently of
EMDB accuracy; validation is performed before the one-pair smoke gate.

## Verified reuse and runtime accounting

Identity ICP/CPD and Go-ICP rows are reused only after the existing dataset and
checkpoint validators check the full 80-row reference against manifests,
input/rotation hashes, method settings, code, package versions and environment.
Their original runtimes and transformations are retained and labelled as prior
measurements. No new Go-ICP search is run. Go-ICP searches full SO(3) via the
axis-angle cube [-3.1416,3.1416]^3 enclosing the pi-radius rotation ball, and
translation [-.05,.05]^3, DT=300, MSE=.001, all 1000 points, trim=0, DT expansion=2.
The existing guarded rotation-cleanup policy (5e-6) remains unchanged. It has
no single local R_init, so its initial angle/refinement time are not applicable.
This global-search reference is not a claim of exact optimality under the
distance-transform approximation and finite stopping tolerance.

For shared methods, registration/refinement runtime is measured separately from
the common initialization. Per-method standalone total = initialization + its
refinement. The common time is shown in each standalone total but is counted
ONCE in the actual experiment ledger below, not summed three times. Reused
historical fit costs are zero new execution cost, yet remain visible in the
method runtime comparison. The boxes show total standalone cost; open circles
show shared refinement-only times, connected to their totals.

```json
{json.dumps(ledger,indent=2)}
```

Identical initial rotations do not imply identical objectives or parameter
spaces: MMD is rotation-only; ICP/Go-ICP estimate translation; CPD also estimates
scale. The PCA/greedy initialization is selected using MMD and may favor it.
Identity-start MMD/ICP/CPD are the complementary starting-rotation control.
Convergence is separate from accuracy; no solver parameters were tuned using
true rotations and no local solver is claimed globally optimal.

## Outputs and preservation

trial_results.csv: 7 conditions/pair, initial/final angles, Frobenius errors,
improvement, split runtimes/provenance, translations/scales, convergence and
native diagnostics. initialization_diagnostics.csv stores each R_init, frames,
signs, initial MMD, initialization angle and PCA/greedy costs once per pair.
aggregated_results.csv, pairwise_results.csv and paired_comparison_summary.csv
provide descriptive summaries. validation_results.json, smoke_validation.json,
checkpoint_validation.json and preservation_check.json document validations.
All figures use actual SciencePlots ['science','no-latex'], 600-dpi PNG and
vector PDF. No old benchmark scripts, manifests, data, CSVs or figures changed.
The common-start computation is not a true-frame perturbation experiment.
No Overleaf files were edited.

Run/resume with the same validated Python environment:
```powershell
& .\\data\\.venv-emdb-baselines\\Scripts\\python.exe -B emdb_shared_initialization_comparison.py
& .\\data\\.venv-emdb-baselines\\Scripts\\python.exe -B emdb_shared_initialization_comparison.py --resume
```
The first completed pair is an incremental smoke gate before the remaining
pairs. Partial method rows and initialization records are atomic checkpoints;
resume verifies signatures and prefixes and skips completed work.
"""
    (output/'REPORT.md').write_text(text,encoding='utf-8')


def run(output,num_pairs=20,resume=False):
    if not 1<=num_pairs<=20:raise ValueError('num_pairs must be between 1 and 20.')
    if not resume and output.exists() and any(output.iterdir()):raise ValueError('Use a new output directory; no previous outputs overwritten.')
    began=perf_counter()
    pairs,reference_signature,_=frozen.preflight(frozen.DEFAULT_MANIFEST,20)
    verified=frozen.load_checkpoint(REFERENCE,pairs,frozen.DEFAULT_MANIFEST,20,reference_signature)
    if len(verified)!=80:raise ValueError('The reference must contain all 20 four-method pairs.')
    saved={(r['pair_id'],r['method']):r for r in verified}
    signature={'reference_signature':reference_signature,'reference_csv_sha256':dataset.sha256(REFERENCE),
               'reference_metadata_sha256':dataset.sha256(REFERENCE.parent/'baseline_run_metadata.json'),
               'new_driver_sha256':dataset.sha256(Path(__file__)),'SciencePlots_version':metadata.version('SciencePlots'),
               'cpd_library_sha256':dataset.sha256(Path(inspect.getfile(core._require_dependency('pycpd').RigidRegistration)))}
    meta_path=output/'experiment_metadata.json'
    trial_path=output/'trial_results.csv';init_path=output/'initialization_diagnostics.csv'
    if resume:
        meta=json.loads(meta_path.read_text())
        if meta['signature']!=signature:raise ValueError('Resume signature differs; cannot mix data/code/settings.')
        rows=dataset.read_manifest(trial_path) if trial_path.exists() else []
        initializations=dataset.read_manifest(init_path) if init_path.exists() else []
        validate_rows(rows,initializations,pairs,saved,num_pairs)
        validation=json.loads((output/'validation_results.json').read_text())
        if not validation['passed']:raise ValueError('Prior convention validation did not pass.')
    else:
        protected=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list((ROOT/'numerical_results').rglob('*'))+list(dataset.DEFAULT_OUTPUT.iterdir())+[dataset.DEFAULT_MAP]
        hashes={str(p.resolve()):dataset.sha256(p) for p in protected if p.is_file() and output.resolve() not in p.resolve().parents
                and p.name not in ('emdb_shared_initialization_comparison.py','test_emdb_shared_initialization.py')}
        output.mkdir(parents=True,exist_ok=True)
        frozen.write_json(output/'protected_file_hashes.json',hashes)
        meta={'signature':signature,'num_pairs':num_pairs,'status':'validating','reference_reused_methods':['icp_identity','cpd_identity','goicp_global'],
              'no_GoICP_searches_run':True,'settings':frozen.method_settings(),'cpd_initialization':'pre-rotation + default incremental CPD',
              'runtime_accounting':'shared init once per pair; standalone totals include overhead; cached runtimes retain provenance'}
        frozen.write_json(meta_path,meta)
        print('Synthetic known-rotation ICP/CPD convention validation...',flush=True)
        validation=synthetic_validation();frozen.write_json(output/'validation_results.json',validation)
        rows,initializations=[],[]
    try:
        for pair in pairs[:num_pairs]:
            completed={r['method_condition'] for r in rows if r['pair_id']==pair['pair_id']}
            if completed==set(EXECUTION):continue
            Xc,Yc,truth,scale=frozen.prepare_pair(pair,frozen.DEFAULT_MANIFEST)
            fingerprints=(frozen.array_hash(Xc),frozen.array_hash(Yc),frozen.array_hash(truth))
            init_record=next((r for r in initializations if r['pair_id']==pair['pair_id']),None)
            if init_record is None:
                R_init,init_record=shared_initialization(Xc,Yc,int(pair['ours_fit_seed']))
                objective=core.RotationMMD(Xc,Yc,CONFIG.sigma)
                init_record.update(pair_id=pair['pair_id'],R_init=json.dumps(R_init.tolist()),ours_fit_seed=int(pair['ours_fit_seed']),
                                   Xc_sha256=fingerprints[0],Yc_sha256=fingerprints[1],normalization_scale=scale,
                                   initial_mmd=objective.evaluate(R_init),initial_angle_error_degrees=core.rotation_angle_error_degrees(truth,R_init))
                initializations.append(init_record);frozen.write_checkpoint(init_path,initializations)
            else:
                R_init=np.array(json.loads(init_record['R_init']));R_init.setflags(write=False)
                if init_record['Xc_sha256']!=fingerprints[0] or init_record['Yc_sha256']!=fingerprints[1]:raise ValueError('Initialization input hash mismatch.')
            core.assert_rotation(R_init)
            init_seconds=float(init_record['initialization_runtime_seconds'])
            for key in EXECUTION:
                if key in completed:continue
                print(f"{pair['pair_id']} {key}: starting {'verified reuse' if key in ('icp_identity','cpd_identity','goicp_global') else 'refinement'}",flush=True)
                reused=key in ('icp_identity','cpd_identity','goicp_global')
                start=R_init if key.endswith('_shared') else np.eye(3)
                if key=='mmd_shared' or key=='mmd_identity':result=mmd_refinement(Xc,Yc,start)
                elif key=='icp_shared':result=icp_refinement(Xc,Yc,start)
                elif key=='cpd_shared':result=cpd_refinement(Xc,Yc,start)
                else:result=cached_result(saved[pair['pair_id'],key.split('_')[0]],key.split('_')[0])
                reproduction=verify_mmd_reproduction(result,saved[pair['pair_id'],'ours']) if key=='mmd_shared' else None
                if fingerprints!=(frozen.array_hash(Xc),frozen.array_hash(Yc),frozen.array_hash(truth)):
                    raise RuntimeError('Shared frozen input or truth was mutated.')
                row=make_row(pair,Xc,Yc,truth,scale,key,result,start,init_seconds,reused,reproduction)
                if key=='mmd_shared':
                    np.testing.assert_allclose(row['final_angle_error_degrees'],float(saved[pair['pair_id'],'ours']['rotation_angle_error_degrees']),atol=1e-7,rtol=0)
                rows.append(row);frozen.write_checkpoint(trial_path,rows)
                print(f"{pair['pair_id']} {key}: angle={row['final_angle_error_degrees']:.9f}, time={row['registration_runtime_seconds']:.3f}s",flush=True)
            validate_rows(rows,initializations,pairs,saved,num_pairs)
            if pair['pair_id']=='pair_000':
                frozen.write_json(output/'smoke_validation.json',{'passed':True,'pair_id':'pair_000','rows':7,
                                  'MMD_reproduced_saved_Ours':True,'synthetic_conventions_passed':True,'no_GoICP_rerun':True})
                print('One-pair smoke passed; continuing with remaining frozen pairs.',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        frozen.write_json(output/'experiment_failure.json',{'exception_type':type(exc).__name__,'message':str(exc),
                          'completed_rows':len(rows),'completed_initializations':len(initializations),'checkpoint':str(trial_path)})
        raise
    serialized=dataset.read_manifest(trial_path)
    validate_rows(serialized,initializations,pairs,saved,num_pairs,complete=True)
    numeric=numeric_rows(serialized)
    summaries=statistics(numeric);frozen.write_checkpoint(output/'aggregated_results.csv',summaries)
    wide,comparisons=pairwise(numeric,initializations)
    frozen.write_checkpoint(output/'pairwise_results.csv',wide);frozen.write_checkpoint(output/'paired_comparison_summary.csv',comparisons)
    ledger={'initialization_computations':len(initializations),
            'common_initialization_seconds_counted_once':sum(float(r['initialization_runtime_seconds']) for r in initializations),
            'new_registration_adapter_seconds':sum(r['execution_runtime_seconds_this_run'] for r in numeric),
            'new_local_refinements':sum(not (r['result_reused'] is True or r['result_reused']=='True') for r in numeric),
            'reused_identity_ICP_CPD_GoICP_rows':sum(r['result_reused'] is True or r['result_reused']=='True' for r in numeric),
            'historical_runtime_not_added_to_new_execution_cost':True}
    ledger['unique_init_plus_new_fit_seconds']=ledger['common_initialization_seconds_counted_once']+ledger['new_registration_adapter_seconds']
    frozen.write_json(output/'runtime_ledger.json',ledger)
    plots(numeric,wide,output)
    # Serialized booleans are made explicit for report counts; numerical results remain unchanged.
    for row in numeric:
        row['solver_converged']=None if row['solver_converged'] in ('',None) else row['solver_converged'] is True or row['solver_converged']=='True'
    write_report(numeric,initializations,summaries,comparisons,ledger,output)
    hashes=json.loads((output/'protected_file_hashes.json').read_text())
    changed=[p for p,value in hashes.items() if dataset.sha256(Path(p))!=value]
    frozen.write_json(output/'preservation_check.json',{'passed':not changed,'protected_files':len(hashes),'changed_files':changed})
    frozen.write_json(output/'checkpoint_validation.json',{'passed':True,'rows':len(rows),'conditions_per_pair':7,
                      'initializations':len(initializations),'all_shared_MMD_rows_reproduce_reference':True,'no_GoICP_searches_run':True,
                      'maximum_reference_R_fro_difference':max(float(r['reference_R_fro_difference']) for r in numeric if r['method_condition']=='mmd_shared')})
    if changed:raise RuntimeError('Protected originals changed; see preservation_check.json.')
    meta.update(status='completed',num_pairs=num_pairs,rows=len(rows),invocation_elapsed_seconds=perf_counter()-began)
    frozen.write_json(meta_path,meta)
    print(f"Completed {len(rows)} rows / {num_pairs} frozen pairs. No Go-ICP reruns. {output}",flush=True)
    return numeric


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--num-pairs',type=int,default=20)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.output_dir.resolve(),args.num_pairs,args.resume)


if __name__=='__main__':
    main()
