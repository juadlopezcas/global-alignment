"""Held-out evaluation of five FIXED saved rotations for all 20 EMDB pairs.

No fitting or new rotations. Reuse the original held-out sampling/preprocessing
protocol; exact float64 Gaussian sums, diagonal-excluding MMD, sigma=1.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t

import diagnose_emdb_mmd_cross_evaluation as training
import diagnose_emdb_heldout_mmd as previous
from research_plot_style import research_style

core, frozen, shared, sampling = training.core, training.frozen, training.shared, training.dataset
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'numerical_results/emdb_heldout_mmd_cross_evaluation'
OLD_OUTPUT = previous.DEFAULT_OUTPUT
DESIGN = previous.DESIGN  # (1000,20), (5000,10), (10000,5), inspected original.
MASTER_SEED = 20261027
SEED_NAMESPACE = 0x484F3230  # 'HO20': distinct from training and prior pair_017.
CANDIDATES = training.CANDIDATES
CONTRASTS = ('icp', 'cpd', 'true', 'init')
LABELS = {'icp': 'ICP', 'cpd': 'CPD', 'true': 'True', 'init': 'Initialization'}
SIGMA = 1.0
LOSS_TOLERANCE = 1e-12
VALIDATION_TOLERANCE = 3e-14


@contextmanager
def evaluation_only_guard():
    """Allow only held-out sampling, never a fitting or rotation-generation call."""
    forbidden = [(core, name) for name in ('run_ours', 'register_point_clouds', 'estimate_pca_frame',
                  'greedy_sign_refinement', 'run_icp', 'run_rigid_cpd', 'run_goicp', 'random_rotation',
                  'validate_goicp_convention', 'clean_goicp_rotation')]
    forbidden += [(shared, name) for name in ('shared_initialization', 'mmd_refinement', 'icp_refinement',
                  'cpd_refinement', 'synthetic_validation', 'run')]
    forbidden += [(sampling, name) for name in ('generate_pairs', 'random_rotation')]
    forbidden += [(frozen, name) for name in ('run_experiment', 'iter_pair_results')]
    forbidden += [(core.TrustRegions, 'run')]
    with ExitStack() as stack:
        guards = [stack.enter_context(patch.object(module, name, side_effect=AssertionError(f'Forbidden fitting/correction: {name}')))
                  for module, name in forbidden]
        yield
        for guard in guards:
            guard.assert_not_called()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sampling_plan(pairs, master_seed=MASTER_SEED):
    """Independent pair/size/repetition streams predeclared before loss evaluation."""
    if master_seed < 0:
        raise ValueError('Seed must be nonnegative.')
    seen = {int(row[key]) for row in pairs for key in sampling.STREAM_NAMES}
    for row in sampling.read_manifest(OLD_OUTPUT / 'sampling_plan.csv'):
        seen.update(int(row[k]) for k in ('source_sampling_seed', 'target_sampling_seed'))
    rows = []
    for n, repetitions in DESIGN:
        for pair_index, pair in enumerate(pairs):
            for repetition in range(repetitions):
                entropy = [master_seed, SEED_NAMESPACE, pair_index, n, repetition]
                children = np.random.SeedSequence(entropy).spawn(2)
                seeds = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]
                if seeds[0] == seeds[1] or any(s in seen for s in seeds):
                    raise ValueError('Seed collision; no adaptive reseeding.')
                seen.update(seeds)
                rows.append({'evaluation_id': f"{pair['pair_id']}_n{n}_rep{repetition:03d}",
                             'pair_id': pair['pair_id'], 'num_points': n, 'repetition': repetition,
                             'planned_repetitions': repetitions, 'master_seed': master_seed,
                             'seed_entropy': json.dumps(entropy), 'source_sampling_seed': seeds[0],
                             'target_sampling_seed': seeds[1]})
    return rows


def blocked_cross_mean(X, Y, chunk_size=512):
    """Exact cdist/exp used by RotationMMD.evaluate, with bounded tile memory."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    sums = []
    for i in range(0, len(X), chunk_size):
        for j in range(0, len(Y), chunk_size):
            block = cdist(X[i:i+chunk_size], Y[j:j+chunk_size], metric='sqeuclidean')
            block *= -.5 / SIGMA**2
            np.exp(block, out=block)
            sums.append(float(block.sum(dtype=np.float64)))
    return math.fsum(sums) / (len(X) * len(Y))


def blocked_within(X, chunk_size=512):
    """Exclude actual kernel diagonal; symmetric off-diagonal tiles counted twice."""
    sums = []
    for i in range(0, len(X), chunk_size):
        for j in range(i, len(X), chunk_size):
            block = cdist(X[i:i+chunk_size], X[j:j+chunk_size], metric='sqeuclidean')
            block *= -.5 / SIGMA**2
            np.exp(block, out=block)
            if i == j:
                np.fill_diagonal(block, 0.)
            sums.append(float(block.sum(dtype=np.float64)) * (1 if i == j else 2))
    return math.fsum(sums) / (len(X) * (len(X)-1))


def evaluate_pair(X, Y, rotations, chunk_size=512):
    """One within term per cloud; rotation-only candidates share exact X/Y arrays."""
    before = (frozen.array_hash(X), frozen.array_hash(Y))
    rh = {k: frozen.array_hash(R) for k, R in rotations.items()}
    within = blocked_within(X, chunk_size) + blocked_within(Y, chunk_size)
    means = {k: blocked_cross_mean(X @ R.T, Y, chunk_size) for k, R in rotations.items()}
    losses = {k: within - 2*v for k, v in means.items()}
    differences = {k: -2*(means['R_'+k] - means['R_mmd']) for k in CONTRASTS}
    for k, d in differences.items():
        np.testing.assert_allclose(d, losses['R_'+k]-losses['R_mmd'], atol=1e-15, rtol=0)
    if not np.isfinite([within, *means.values(), *losses.values(), *differences.values()]).all():
        raise ValueError('Nonfinite MMD.')
    if before != (frozen.array_hash(X), frozen.array_hash(Y)) or rh != {k: frozen.array_hash(R) for k, R in rotations.items()}:
        raise ValueError('Candidate or shared input changed.')
    return within, means, losses, differences


def load_inputs():
    """Reuse the complete provenance and CPD composition validators from training."""
    pairs, rows, initializations, prepared, signature = training.load_validated_inputs()
    inventory = json.loads((training.DEFAULT_OUTPUT / 'output_manifest.json').read_text())
    for entry in inventory['files']:
        if sampling.sha256(Path(entry['path'])) != entry['sha256']:
            raise ValueError('Training cross-evaluation inventory mismatch.')
    candidate_rows = sampling.read_manifest(training.DEFAULT_OUTPUT / 'candidate_mmd_values.csv')
    train_rows = sampling.read_manifest(training.DEFAULT_OUTPUT / 'pairwise_mmd_differences.csv')
    by_pair = {r['pair_id']: r for r in train_rows}
    initial = {r['pair_id']: r for r in initializations}
    rotations, identities = {}, []
    for pair in pairs:
        pid = pair['pair_id']
        rr, lookup, composition = training.saved_candidates(pair, initial[pid], rows, prepared[pid])
        rotations[pid] = rr
        for name, R in rr.items():
            saved = next(r for r in candidate_rows if r['pair_id']==pid and r['candidate']==name)
            np.testing.assert_array_equal(R, json.loads(saved['R']))
            if frozen.array_hash(R) != saved['rotation_sha256']:
                raise ValueError('Saved rotation hash differs.')
            actual_angle = core.rotation_angle_error_degrees(rr['R_true'], R)
            np.testing.assert_allclose(actual_angle, float(saved['angular_error_degrees']), atol=1e-10, rtol=0)
            identities.append({'pair_id': pid, 'candidate': name, 'R': R.tolist(),
                               'rotation_sha256': frozen.array_hash(R),
                               'angular_error_degrees': actual_angle, 'training_mmd': float(saved['empirical_mmd']),
                               'source_condition': saved['source_condition'], 'cpd_composition_max_residual': composition})
        for key in CONTRASTS:
            expected = (float(by_pair[pid]['L_'+key])-float(by_pair[pid]['L_mmd']))
            if key != 'init':
                np.testing.assert_allclose(expected, float(by_pair[pid]['Delta_'+key]), atol=1e-15, rtol=0)
    meta = json.loads((OLD_OUTPUT/'run_metadata.json').read_text())
    if (meta['status'] != 'completed' or meta['completed_repetitions'] != 35 or meta['sigma'] != SIGMA
            or meta['diagnostic_source_sha256'] != sampling.sha256(Path(previous.__file__))
            or meta['map_sha256'] != pairs[0]['map_sha256']):
        raise ValueError('Earlier held-out protocol is missing, changed, or incomplete.')
    for name, oldname in (('R_mmd', 'ours'), ('R_true', 'true')):
        np.testing.assert_array_equal(rotations['pair_017'][name], meta['candidates']['rotations'][oldname])
    return pairs, rotations, identities, by_pair, signature


def validate_numerics(population, entry, rotations, chunk_size):
    X, Y, details = previous.sample_pair(population, entry, rotations['R_true'])
    X2, Y2, details2 = previous.sample_pair(population, entry, rotations['R_true'])
    np.testing.assert_array_equal(X, X2); np.testing.assert_array_equal(Y, Y2)
    assert details == details2
    _, _, losses, deltas = evaluate_pair(X, Y, rotations, chunk_size)
    dense = core.RotationMMD(X, Y, SIGMA)
    checks = []
    for name, R in rotations.items():
        original = dense.evaluate(R)
        full = core.mmd_squared(X @ R.T, Y, SIGMA)
        residual = max(abs(original-losses[name]), abs(full-losses[name]))
        if residual > VALIDATION_TOLERANCE:
            raise ValueError('Blockwise/dense MMD mismatch.')
        checks.append({'candidate': name, 'blocked_mmd': losses[name], 'dense_mmd': original,
                       'independent_mmd': full, 'max_absolute_discrepancy': residual})
    chunk_checks = []
    for chunk in (257, 1024):
        _, _, other, ds = evaluate_pair(X, Y, rotations, chunk)
        residual = max(abs(other[k]-losses[k]) for k in losses)
        residual = max(residual, max(abs(ds[k]-deltas[k]) for k in deltas))
        if residual > VALIDATION_TOLERANCE:
            raise ValueError('Chunk-size mismatch.')
        chunk_checks.append({'chunk_size': chunk, 'max_absolute_discrepancy': residual})
    _, _, _, identical = evaluate_pair(X[:31], Y[:31], {k: rotations['R_true'] for k in rotations}, 17)
    assert all(d == 0 for d in identical.values())
    return {'passed': True, 'dense_checks_n1000': checks, 'chunk_checks': chunk_checks,
            'same_seed_exact_arrays_and_hashes': True, 'identical_rotation_deltas_exactly_zero': True,
            'numerical_tolerance': VALIDATION_TOLERANCE}


def validate_previous(population, rotations, output, chunk_size):
    """Replay ALL 35 original pair_017 streams, with the two exact fixed matrices."""
    meta = json.loads((OLD_OUTPUT/'run_metadata.json').read_text())
    plan = previous.sampling_plan(meta['master_seed'])
    saved_plan = sampling.read_manifest(OLD_OUTPUT/'sampling_plan.csv')
    old_rows = sampling.read_manifest(OLD_OUTPUT/'heldout_trial_results.csv')
    if len(plan) != 35 or len(old_rows) != 35 or len(saved_plan) != 35:
        raise ValueError('Earlier held-out rows missing.')
    replay = []
    for entry, saved, old in zip(plan, saved_plan, old_rows):
        if any(str(v) != saved[k] for k, v in entry.items()) or old['evaluation_id'] != entry['evaluation_id']:
            raise ValueError('Earlier sampling plan cannot be reconstructed exactly.')
        X, Y, details = previous.sample_pair(population, entry, rotations['R_true'])
        for key, value in details.items():
            if str(value) != old[key]:
                raise ValueError(f'Earlier sample/preprocessing mismatch: {key}')
        ours = blocked_cross_mean(X @ rotations['R_mmd'].T, Y, chunk_size)
        truth = blocked_cross_mean(X @ rotations['R_true'].T, Y, chunk_size)
        new_delta = -2*(ours-truth)  # Earlier Ours-True, opposite to new True-MMD.
        residual = abs(new_delta-float(old['Delta_ours_true']))
        if residual > VALIDATION_TOLERANCE:
            raise ValueError('Earlier pair_017 result does not reproduce.')
        replay.append({**entry, **details, 'earlier_ours_minus_true': float(old['Delta_ours_true']),
                       'replayed_ours_minus_true': new_delta, 'absolute_discrepancy': residual})
    shared.checkpoint(output/'pair_017_legacy_replay.csv', replay)
    return {'passed': True, 'repetitions_replayed': 35, 'exact_original_seeds_and_samples': True,
            'R_mmd_R_true_identical': True, 'same_estimator_sigma_sampling_preprocessing': True,
            'max_absolute_delta_discrepancy': max(r['absolute_discrepancy'] for r in replay),
            'old_difference_sign': 'Ours-True; negate for new True-MMD',
            'GoICP_not_included': True}


def summarize(rows, training_rows, identities):
    """First aggregate paired differences WITHIN each original fitted rotation set."""
    result = []
    angles = {(r['pair_id'], r['candidate']): r['angular_error_degrees'] for r in identities}
    for n, planned in DESIGN:
        for pid in sorted(training_rows):
            group = [r for r in rows if int(r['num_points'])==n and r['pair_id']==pid]
            if not group:
                continue
            for key in CONTRASTS:
                v = np.array([float(r['Delta_'+key]) for r in group], dtype=np.float64)
                count = len(v); mean = float(v.mean())
                std = float(v.std(ddof=1)) if count>1 else float('nan')
                half = float(student_t.ppf(.975, count-1)*std/np.sqrt(count)) if count>1 else float('nan')
                lo, hi = mean-half, mean+half
                train_delta = float(training_rows[pid]['L_'+key])-float(training_rows[pid]['L_mmd'])
                preference = training.classify_alternative(mean)
                original = training.classify_alternative(train_delta)
                clear = 'alternative_lower' if hi < -LOSS_TOLERANCE else 'MMD_lower' if lo > LOSS_TOLERANCE else 'uncertain'
                result.append({'pair_id': pid, 'num_points': n, 'comparison': key, 'repetitions': count,
                               'planned_repetitions': planned, 'complete': count==planned,
                               'mean_delta': mean, 'sample_std_delta': std, 'median_delta': float(np.median(v)),
                               'min_delta': float(v.min()), 'max_delta': float(v.max()),
                               'ci95_mean_lower': lo, 'ci95_mean_upper': hi,
                               'ci_excludes_zero': bool(lo>0 or hi<0), 'ci_clear_preference': clear,
                               'mean_preference': preference, 'training_delta': train_delta,
                               'training_preference': original,
                               'training_preference_reversed': preference != original and 'numerical_tie' not in (preference, original),
                               'alternative_preferred_repetitions': int(np.sum(v < -LOSS_TOLERANCE)),
                               'MMD_preferred_repetitions': int(np.sum(v > LOSS_TOLERANCE)),
                               'numerical_tie_repetitions': int(np.sum(np.abs(v)<=LOSS_TOLERANCE)),
                               'alternative_angle_degrees': angles[(pid, 'R_'+key)],
                               'MMD_angle_degrees': angles[(pid, 'R_mmd')],
                               'ci_method': 'pointwise paired Student-t, df=repetitions-1; no multiplicity correction',
                               'sigma': SIGMA})
    return result


def sample_size_analysis(summaries):
    """Descriptive across fitted pairs; never pool repetitions into an iid CI."""
    rows = []
    for n, repetitions in DESIGN:
        for key in CONTRASTS:
            group = [r for r in summaries if r['num_points']==n and r['comparison']==key and r['complete']]
            if not group:
                continue
            v = np.array([r['mean_delta'] for r in group])
            rows.append({'num_points': n, 'comparison': key, 'completed_pairs': len(group),
                         'repetitions_per_pair': repetitions, 'mean_of_pair_means': float(v.mean()),
                         'sample_std_across_pair_means': float(v.std(ddof=1)) if len(v)>1 else float('nan'),
                         'median_pair_mean': float(np.median(v)), 'min_pair_mean': float(v.min()), 'max_pair_mean': float(v.max()),
                         'alternative_mean_preferred_pairs': sum(r['mean_preference']=='alternative_lower' for r in group),
                         'MMD_mean_preferred_pairs': sum(r['mean_preference']=='MMD_lower' for r in group),
                         'numerical_tie_pairs': sum(r['mean_preference']=='numerical_tie' for r in group),
                         'clear_alternative_pairs': sum(r['ci_clear_preference']=='alternative_lower' for r in group),
                         'clear_MMD_pairs': sum(r['ci_clear_preference']=='MMD_lower' for r in group),
                         'uncertain_pairs': sum(r['ci_clear_preference']=='uncertain' for r in group),
                         'ci_excludes_zero_pairs': sum(r['ci_excludes_zero'] for r in group),
                         'training_preference_reversed_pairs': sum(r['training_preference_reversed'] for r in group),
                         'summary_type': 'descriptive across fixed fitted pairs; no pooled CI'})
    return rows


def save_plots(summaries, size_rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    primary = [r for r in summaries if r['num_points']==1000 and r['complete']]
    if len(primary) != 80:
        return
    def save(fig, name):
        for ext in ('png', 'pdf'):
            fig.savefig(output/f'{name}.{ext}', **({'dpi': 600} if ext=='png' else {}))
        plt.close(fig)
    with research_style():
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
        for ax, key in zip(axes, CONTRASTS[:3]):
            group = [r for r in primary if r['comparison']==key]
            x = np.arange(20)
            means = np.array([r['mean_delta'] for r in group])
            errors = np.array([[r['mean_delta']-r['ci95_mean_lower'] for r in group],
                               [r['ci95_mean_upper']-r['mean_delta'] for r in group]])
            ax.plot(x-.1, [1e6*r['training_delta'] for r in group], 's', color='#777777', ms=4, label='Original training')
            ax.errorbar(x+.1, 1e6*means, yerr=1e6*errors, fmt='o', ms=4, color='#0C5DA5', capsize=2,
                        label='Held-out mean, pointwise 95% t CI')
            ax.axhline(0, color='black', lw=.8)
            ax.set(ylabel=rf'$\Delta_{{\mathrm{{{LABELS[key]}}}}}\times10^6$', title=f'{LABELS[key]} minus MMD')
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(ncol=2, fontsize=9)
        axes[-1].set_xticks(range(20), [f'{i:03d}' for i in range(20)], rotation=45)
        axes[-1].set_xlabel('Original frozen pair index')
        fig.suptitle('Training versus independent held-out MMD; n=1000, 20 repetitions/pair\nNegative difference favors the alternative fixed rotation', fontsize=12)
        save(fig, 'training-versus-heldout-mmd')
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
        keys = CONTRASTS[:3]; x = np.arange(3); bottom = np.zeros(3)
        categories = [('clear_alternative_pairs', 'Alternative lower; CI excludes zero', '#00B945'),
                      ('uncertain_pairs', 'Uncertain; CI does not establish a preference', '#AAAAAA'),
                      ('clear_MMD_pairs', 'MMD lower; CI excludes zero', '#FF9500')]
        for field, label, color in categories:
            values = [next(r[field] for r in size_rows if r['num_points']==1000 and r['comparison']==k) for k in keys]
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=.6)
            for i, value in enumerate(values):
                if value: ax.text(i, bottom[i]+value/2, str(value), ha='center', va='center')
            bottom += values
        ax.set_xticks(x, ['ICP versus MMD', 'CPD versus MMD', 'True versus MMD'])
        ax.set(ylim=(0, 23), yticks=range(0, 21, 5), ylabel='Number of original fitted pairs',
               title='Held-out MMD preference: n=1000, 20 repetitions/pair')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), fontsize=8)
        save(fig, 'heldout-preference-counts')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained', sharex=True)
        available = [n for n, _ in DESIGN if sum(r['num_points']==n and r['complete'] for r in summaries)==80]
        for ax, key in zip(axes, keys):
            for pid in [f'pair_{i:03d}' for i in range(20)]:
                group = [next(r for r in summaries if r['pair_id']==pid and r['num_points']==n and r['comparison']==key) for n in available]
                ax.plot(range(len(available)), [1e6*r['mean_delta'] for r in group], color='#0C5DA5', alpha=.25, lw=.7)
            groups = [next(r for r in size_rows if r['num_points']==n and r['comparison']==key) for n in available]
            ax.plot(range(len(available)), [1e6*r['mean_of_pair_means'] for r in groups], 'ko-', label='Mean of 20 pair means')
            ax.axhline(0, color='black', lw=.7)
            ax.set_xticks(range(len(available)), [f'{n}\n{dict(DESIGN)[n]} reps/pair' for n in available])
            ax.set(title=f'{LABELS[key]} minus MMD', xlabel='Points per evaluation cloud', ylabel=r'Mean paired MMD difference $\times10^6$')
            ax.grid(axis='y', alpha=.2)
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        axes[0].legend(fontsize=8)
        fig.suptitle('Fixed rotations evaluated at larger n; faint lines are individual fitted pairs\nNo registration refitting; across-pair summaries are descriptive', fontsize=11)
        save(fig, 'evaluation-sample-size-effect')


def write_report(rows, summaries, size_rows, validation, output, complete):
    lines = ['# Held-out MMD cross-evaluation: EMD-9515, all 20 fixed rotation sets', '',
             f'Status: {"completed" if complete else "checkpoint; some settings incomplete"}. {len(rows)}/700 evaluation repetitions; five fixed rotations each.', '',
             'No registration, initialization, rotation correction or bandwidth tuning. Optional Go-ICP omitted.', '',
             '## Protocol', '',
             'Reuse diagnose_emdb_heldout_mmd.sample_pair unchanged: threshold >0.035, 78351 eligible voxels,',
             'coordinates (x,y,z)*voxel_size.x, uniform without replacement within each cloud, independent clouds',
             'with overlap allowed; %.4f ASCII round-trip before rotation. Y=B@R_true.T; independently center A/Y',
             'and divide both by one common maximum centered radius, once per repetition. This sample-dependent',
             'centering/radius is shared by ALL candidates; no scale/translation from a fit is applied.', '',
             'Each fitted pair/size/repetition has two distinct predeclared streams: SeedSequence([20261027,',
             f'{SEED_NAMESPACE}, pair_index, n, repetition]).spawn(2), generate_state(1,uint64), default_rng.',
             'The plan is saved before evaluation. Streams are distinct from training and legacy evaluation seeds.',
             'Evaluation clouds are NOT reused across fitted pairs. Independent draws may overlap in voxels.', '',
             'Exact float64 Gaussian kernel sigma=1; diagonal-excluding within terms computed once/cloud using',
             'symmetric tiles; cross means computed with the same cdist/exp as RotationMMD.evaluate.',
             'Losses are unscaled and may be negative. Delta=alternative minus MMD; within terms cancel in deltas.',
             f'Numerical preference tolerance: {LOSS_TOLERANCE}; validation tolerance: {VALIDATION_TOLERANCE}.', '',
             '## Primary results', '',
             '| Comparison | Mean prefers alternative | Mean prefers MMD | Training sign reversals | Clear alternative | Clear MMD | Uncertain | CI excludes zero |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    primary = [r for r in size_rows if r['num_points']==1000]
    for r in primary:
        lines.append(f"| {LABELS[r['comparison']]} - MMD | {r['alternative_mean_preferred_pairs']}/20 | {r['MMD_mean_preferred_pairs']}/20 | {r['training_preference_reversed_pairs']}/20 | {r['clear_alternative_pairs']}/20 | {r['clear_MMD_pairs']}/20 | {r['uncertain_pairs']}/20 | {r['ci_excludes_zero_pairs']}/20 |")
    lines += ['', '## Sample sizes (descriptive across fitted pairs)', '',
              '| n | comparison | completed pairs | mean of pair means | median | range | alternative-preferred means | clear alternative CIs |',
              '|---:|---|---:|---:|---:|---|---:|---:|']
    for r in size_rows:
        lines.append(f"| {r['num_points']} | {LABELS[r['comparison']]} - MMD | {r['completed_pairs']}/20 | {r['mean_of_pair_means']:.12e} | {r['median_pair_mean']:.12e} | [{r['min_pair_mean']:.12e}, {r['max_pair_mean']:.12e}] | {r['alternative_mean_preferred_pairs']} | {r['clear_alternative_pairs']} |")
    lines += ['', '## Pair_017', '',
              'The legacy fixed R_mmd and R_true are bitwise identical to the new candidates. All 35 old seeds',
              'were replayed separately; source/target XYZ hashes, normalized hashes, radius and overlap match.',
              f"Maximum absolute earlier Ours-True delta discrepancy: {validation['legacy_pair_017']['max_absolute_delta_discrepancy']:.17g}.",
              'Legacy Ours-True is NEGATED to compare with the new True-MMD sign convention.', '',
              '| n | comparison | training delta | held-out mean | pointwise 95% CI | reversal |',
              '|---:|---|---:|---:|---|---|']
    for r in summaries:
        if r['pair_id']=='pair_017':
            lines.append(f"| {r['num_points']} | {LABELS[r['comparison']]} - MMD | {r['training_delta']:.12e} | {r['mean_delta']:.12e} | [{r['ci95_mean_lower']:.12e}, {r['ci95_mean_upper']:.12e}] | {r['training_preference_reversed']} |")
    alt = sum(r['alternative_mean_preferred_pairs'] for r in primary if r['comparison'] in CONTRASTS[:3])
    mmd = sum(r['MMD_mean_preferred_pairs'] for r in primary if r['comparison'] in CONTRASTS[:3])
    interpretation = ('Training preference usually reverses on held-out clouds, supporting finite-sample training-loss overfitting/objective mismatch for these fixed fits.' if alt>mmd else
                      'Held-out means usually retain the MMD preference; investigate sampling law, normalization, weak identifiability and bandwidth in separate studies.' if mmd>alt else
                      'The mean preferences are mixed; there is no single dominant explanation.')
    lines += ['', '## Interpretation and uncertainty', '', interpretation,
              'Inspect the per-comparison counts and intervals above; mean signs alone do not establish a clear preference.',
              'CIs use the paired difference from each whole independent evaluation repetition, conditional on the fixed fitted rotations.',
              'Pointwise Student-t: mean +/- t_(r-1,.975)*sample_SD/sqrt(r); iid repetitions under the specified sampling/preprocessing law,',
              'and approximately normal mean (exact coverage would require normal repetition differences). At r=5/10, normality is not',
              'assured and interval coverage is approximate. No multiplicity adjustment; the CI counts are descriptive, not a familywise claim.',
              'No candidate variances are combined as independent. No pooled repetition-level inference across fitted pairs is used.',
              'These intervals are conditional Monte Carlo uncertainty, not uncertainty over a population of trained registrations.',
              'Failure to exclude zero is not equivalence. Larger n also changes the random radius and centering, and repetitions decrease',
              'with n; no rate or improved registration accuracy follows from these fixed-rotation evaluations.',
              'This does not establish population optimality/consistency, local/global solver optimality or universal method superiority.',
              'A next study could increase independent repetitions for unresolved pairs with the same predeclared protocol; do not tune using truth.', '',
              '## Numerical checks, preservation and outputs', '',
              f"Dense/blockwise maximum discrepancy: {max(r['max_absolute_discrepancy'] for r in validation['dense_checks_n1000']):.17g}.",
              'Candidate source conditions, hashes, strict SO(3), CPD total rotation composition and original source inventories verified.',
              'No candidates projected, transposed anew, inverted or otherwise corrected. All fit/TrustRegions/correction calls guarded.',
              'Float64 accumulation uses math.fsum of tile sums. All five candidates share identical read-only evaluation arrays.',
              'All earlier files are hash-protected. See preservation_check.json and output_manifest.json for exact paths/checksums.',
              'Repetition CSV records all five absolute losses, four raw deltas, cross means, within sum, seeds, hashes and runtime.',
              'Per-pair summaries contain fixed angular errors, sample SDs, CIs, repetition preferences and training sign changes.',
              'All figures use actual SciencePlots science/no-latex, linear signed axes, 600-dpi PNG and vector PDF. Display-only x1e6.',
              'Existing experiments and Overleaf are untouched.', '',
              '## Completion by setting', '']
    for n, reps in DESIGN:
        count = sum(int(r['num_points'])==n for r in rows)
        lines.append(f'- n={n}: {count}/{20*reps} repetitions ({reps}/pair planned); {"complete" if count==20*reps else "INCOMPLETE"}.')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def checkpoint_rows(path, rows):
    """Hash each CSV row's serialized fields; atomically fsync/replace the CSV."""
    serialized = []
    for row in rows:
        r = {k: str(v) for k, v in row.items() if k!='row_sha256'}
        r['row_sha256'] = digest_json(r)
        serialized.append(r)
    shared.checkpoint(path, serialized)


def load_rows(path, plan, candidate_hash):
    if not path.exists():
        return []
    rows = sampling.read_manifest(path)
    if len(rows)>len(plan):
        raise ValueError('Too many checkpoint rows.')
    for r, entry in zip(rows, plan):
        if digest_json({k:v for k,v in r.items() if k!='row_sha256'}) != r['row_sha256']:
            raise ValueError('Checkpoint row checksum mismatch.')
        if any(r[k]!=str(v) for k,v in entry.items()) or r['candidate_set_sha256'] != candidate_hash:
            raise ValueError('Checkpoint ordering, plan, or candidate identity mismatch.')
        values = [float(r['L_'+name[2:]]) for name in CANDIDATES] + [float(r['Delta_'+k]) for k in CONTRASTS]
        if not np.isfinite(values).all() or float(r['sigma'])!=SIGMA:
            raise ValueError('Nonfinite/invalid checkpoint.')
        for key in CONTRASTS:
            np.testing.assert_allclose(float(r['Delta_'+key]), float(r['L_'+key])-float(r['L_mmd']), atol=1e-15, rtol=0)
    return rows


def protect_files(output):
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.md')) + list((ROOT/'numerical_results').rglob('*'))
    paths += list(sampling.DEFAULT_OUTPUT.iterdir()) + [sampling.DEFAULT_MAP]
    return {str(p.resolve()): sampling.sha256(p) for p in paths if p.is_file()
            and output.resolve() not in p.resolve().parents
            and p.name not in ('diagnose_emdb_heldout_mmd_cross_evaluation.py', 'test_emdb_heldout_mmd_cross_evaluation.py')}


def verify_protected(hashes):
    changed = [p for p, digest in hashes.items() if not Path(p).exists() or sampling.sha256(Path(p))!=digest]
    if changed:
        raise ValueError(f'Protected files changed: {changed}')


def refresh_outputs(rows, train_rows, identities, validation, output, complete):
    summaries = summarize(rows, train_rows, identities)
    sizes = sample_size_analysis(summaries)
    shared.checkpoint(output/'per_pair_heldout_summaries.csv', summaries)
    shared.checkpoint(output/'training_versus_heldout.csv', summaries)
    shared.checkpoint(output/'sample_size_analysis.csv', sizes)
    save_plots(summaries, sizes, output)
    write_report(rows, summaries, sizes, validation, output, complete)
    return summaries, sizes


def run(output, resume=False, chunk_size=512, through_size=10000):
    if chunk_size<1 or through_size not in dict(DESIGN):
        raise ValueError('Invalid chunk size or final sample size.')
    began = perf_counter()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Output exists; use --resume, never overwrite previous results.')
    if resume and not (output/'evaluation_protocol.json').exists():
        raise ValueError('Resume metadata missing.')
    output.mkdir(parents=True, exist_ok=True)
    # The original checkpoint validator rechecks the old, stored Go-ICP cleanup
    # policy for provenance only. No Go-ICP candidate is loaded/evaluated here.
    # Permit that existing validator under its original no-fitting guard, then
    # prohibit every cleanup call throughout the actual held-out evaluation.
    with training.evaluation_only_guard():
        inputs = load_inputs()
    with evaluation_only_guard():
        pairs, rotations, identities, train_rows, source_signature = inputs
        plan = sampling_plan(pairs)
        candidate_hash = digest_json(identities)
        dependencies = [Path(__file__), Path(previous.__file__), Path(training.__file__),
                        Path(core.__file__), Path(frozen.__file__), Path(shared.__file__), Path(sampling.__file__), ROOT/'research_plot_style.py']
        signature = {'design': DESIGN, 'master_seed': MASTER_SEED, 'seed_namespace': SEED_NAMESPACE, 'sigma': SIGMA,
                     'chunk_size': chunk_size, 'candidate_set_sha256': candidate_hash,
                     'source_signature': source_signature,
                     'source_sha256': {str(p): sampling.sha256(p) for p in dependencies},
                     'training_results_sha256': sampling.sha256(training.DEFAULT_OUTPUT/'pairwise_mmd_differences.csv'),
                     'old_results_sha256': sampling.sha256(OLD_OUTPUT/'heldout_trial_results.csv'),
                     'versions': {n: metadata.version(n) for n in ('numpy','scipy','mrcfile','matplotlib','SciencePlots','threadpoolctl')}}
        # Normalize tuples through JSON so resumed comparisons are representation independent.
        signature = json.loads(json.dumps(signature))
        if resume:
            meta = json.loads((output/'evaluation_protocol.json').read_text())
            if meta['signature'] != signature:
                raise ValueError('Resume source/data/settings/candidates changed.')
            hashes = json.loads((output/'protected_file_hashes.json').read_text())
            verify_protected(hashes)
            saved_plan = sampling.read_manifest(output/'sampling_plan.csv')
            if saved_plan != [{k:str(v) for k,v in r.items()} for r in plan]:
                raise ValueError('Saved plan changed.')
        else:
            hashes = protect_files(output)
            frozen.write_json(output/'protected_file_hashes.json', hashes)
            frozen.write_json(output/'fixed_candidates.json', {'sha256': candidate_hash, 'candidates': identities})
            shared.checkpoint(output/'sampling_plan.csv', plan)
            meta = {'signature': signature, 'status': 'running', 'completed_repetitions': 0, 'planned_repetitions': 700,
                    'protocol': 'reuse previous.sample_pair: independent uniform voxel draws; %.4f; B@R_true.T; center separately/common radius once',
                    'CI': 'pointwise Student-t on independent paired repetition deltas; conditional on fixed fits; approximate normal mean; no multiplicity correction',
                    'no_fitting': True, 'GoICP_included': False, 'shared_evaluation_across_pairs': False,
                    'thread_environment': {k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
            frozen.write_json(output/'evaluation_protocol.json', meta)
        population, voxel = sampling.load_population(sampling.DEFAULT_MAP, sampling.contour_threshold(pairs[0]), 10000)
        if len(population)!=int(pairs[0]['num_eligible_voxels']) or voxel!=float(pairs[0]['voxel_size_used']):
            raise ValueError('Eligible population or coordinate conversion differs.')
        if resume and (output/'numerical_validation.json').exists():
            validation = json.loads((output/'numerical_validation.json').read_text())
            if not validation['passed']:
                raise ValueError('Saved validation failed.')
        else:
            print('Validate n=1000 dense/blockwise; then replay all 35 legacy pair_017 streams.', flush=True)
            validation = validate_numerics(population, plan[0], rotations[plan[0]['pair_id']], chunk_size)
            validation['legacy_pair_017'] = validate_previous(population, rotations['pair_017'], output, chunk_size)
            validation.update(all_candidates_SO3_and_CPD_composition_verified=True, candidates_unmodified=True,
                              no_fitting_calls=True, sampling_protocol_reused=True, sigma=SIGMA)
            frozen.write_json(output/'numerical_validation.json', validation)
        rows = load_rows(output/'heldout_repetition_results.csv', plan, candidate_hash)
        resumed_count = len(rows)
        if rows:
            X, Y, detail = previous.sample_pair(population, plan[0], rotations[plan[0]['pair_id']]['R_true'])
            _, _, losses, delta = evaluate_pair(X, Y, rotations[plan[0]['pair_id']], chunk_size)
            if any(str(v)!=rows[0][k] for k,v in detail.items()):
                raise ValueError('Resumed sample replay differs.')
            for k, v in losses.items():
                if float(rows[0]['L_'+k[2:]]) != v:
                    raise ValueError('Resumed loss replay not bitwise identical.')
        seen_hashes = {r[k] for r in rows for k in ('X_xyz_sha256','Z_xyz_sha256')}
        all_candidate_hashes = {pid: {k:frozen.array_hash(v) for k,v in rr.items()} for pid,rr in rotations.items()}
        print(f'Validated inputs; resume={resumed_count}/700. Exact block size={chunk_size}.', flush=True)
        for entry in plan[len(rows):]:
            if entry['num_points']>through_size:
                break
            start = perf_counter(); rr = rotations[entry['pair_id']]
            X, Y, details = previous.sample_pair(population, entry, rr['R_true'])
            if details['X_xyz_sha256'] in seen_hashes or details['Z_xyz_sha256'] in seen_hashes:
                raise ValueError('Duplicate evaluation cloud; no adaptive redrawing.')
            seen_hashes.update((details['X_xyz_sha256'], details['Z_xyz_sha256']))
            kernel_start = perf_counter()
            within, means, losses, deltas = evaluate_pair(X, Y, rr, chunk_size)
            row = {**entry, **details, 'candidate_set_sha256': candidate_hash, 'within_sum': within,
                   **{'cross_mean_'+k[2:]:v for k,v in means.items()}, **{'L_'+k[2:]:v for k,v in losses.items()},
                   **{'Delta_'+k:v for k,v in deltas.items()}, 'sigma': SIGMA, 'chunk_size': chunk_size,
                   'kernel_runtime_seconds': perf_counter()-kernel_start, 'runtime_seconds': perf_counter()-start}
            pending = rows+[row]
            checkpoint_rows(output/'heldout_repetition_results.csv', pending)
            rows.append(row)  # Only after durable commit.
            if len(rows)%20==0 or len(rows)==700:
                print(f"{len(rows)}/700 {entry['evaluation_id']}; kernel {row['kernel_runtime_seconds']:.3f}s; elapsed {perf_counter()-began:.1f}s", flush=True)
            if len(rows) in (400,600,700):
                refresh_outputs(rows, train_rows, identities, validation, output, len(rows)==700)
                meta.update(status='completed' if len(rows)==700 else 'running', completed_repetitions=len(rows))
                frozen.write_json(output/'evaluation_protocol.json', meta)
                verify_protected(hashes)
                if len(rows)==400:
                    primary = [r for r in rows if int(r['num_points'])==1000]
                    median = float(np.median([float(r['kernel_runtime_seconds']) for r in primary]))
                    print(f'Primary complete. Rough remaining kernel estimate: {median*(25*200+100*100):.1f}s; proceed with checkpointed exact sums.', flush=True)
        complete = len(rows)==700
        summaries, sizes = refresh_outputs(rows, train_rows, identities, validation, output, complete)
        verify_protected(hashes)
        if all_candidate_hashes != {pid:{k:frozen.array_hash(v) for k,v in rr.items()} for pid,rr in rotations.items()}:
            raise ValueError('Fixed rotations changed.')
        frozen.write_json(output/'preservation_check.json', {'passed':True,'protected_files':len(hashes),
                          'changed_files':[], 'all_fixed_candidates_unchanged':True, 'no_registration_calls':True})
        elapsed = perf_counter()-began
        meta.update(status='completed' if complete else 'partial_through_requested_size', completed_repetitions=len(rows),
                    resumed_rows=resumed_count, invocation_elapsed_seconds=elapsed,
                    evaluation_repetition_runtime_sum_seconds=sum(float(r['runtime_seconds']) for r in rows))
        frozen.write_json(output/'evaluation_protocol.json', meta)
        frozen.write_json(output/'output_manifest.json', {'directory':str(output), 'files':[
            {'path':str(p.resolve()), 'sha256':sampling.sha256(p)} for p in sorted(output.iterdir())
            if p.is_file() and p.name!='output_manifest.json']})
        print(f"{meta['status']}: {len(rows)}/700 repetitions; this invocation {elapsed:.2f}s; {output}", flush=True)
        return rows, summaries, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=512)
    parser.add_argument('--through-size', type=int, choices=[1000,5000,10000], default=10000)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.resume, args.chunk_size, args.through_size)


if __name__=='__main__':
    main()
