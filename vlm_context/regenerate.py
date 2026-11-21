"""Regenerate manuscript numbers and plots from the bundled numerical evidence.

This path uses no model, dataset download, or fitting. Saved per-example answers
and scores are reduced on CPU, including paired image bootstrap draws with the
recorded settings. Archived summaries serve as regression references. Every
numeric input and generated table/plot is fingerprinted in manifest.json.
"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

from .io import read_json, require, sha256, source_path

Q05_ARMS = (
    ('frozen', 'Frozen', 'None'),
    ('connector_natural', 'Connector', 'Natural'),
    ('connector_episode_remap', 'Connector', 'Episodic'),
    ('late_natural', 'Sparse late', 'Natural'),
    ('late_episode_remap', 'Sparse late', 'Episodic'),
    ('early_episode_remap', 'Sparse early', 'Episodic'),
    ('middle_episode_remap', 'Sparse middle', 'Episodic'),
    ('matched_all_depth_episode_remap', 'Matched all-depth', 'Episodic'),
    ('historical_capped_all_episode_remap', 'Capped language', 'Episodic'),
    ('uncapped_all_episode_remap', 'Uncapped language', 'Episodic'),
)
CORE_ARMS = (
    ('frozen_joint', 'Frozen'),
    ('joint_uncapped_r16_episode_remap', 'Joint'),
    ('late_parameter_matched_r34_35_episode_remap', 'Matched late'),
)
TASKS = (('shapes4', 'Shapes4'), ('cifar4', 'CIFAR4'), ('fashion4', 'Fashion4'))
Q05_KEY = ('task', 'arm', 'condition', 'k', 'episode_seed', 'fit_seed', 'scorer')


def aggregate_q05(records):
    """Rebuild whole-panel confusion/accuracy diagnostics from per-image rows.

    Each image/context/checkpoint tuple must occur exactly once. Invalid outputs
    retain a separate confusion column and remain in accuracy denominators.
    The returned cells preserve each scorer and fitting/context replicate.
    """
    cells, seen = {}, set()
    for row in records:
        identity = tuple(row[k] for k in Q05_KEY[:-1]) + (row['image_id'],)
        require(identity not in seen, 'Duplicate Q05 image/context/checkpoint row')
        seen.add(identity)
        classes = len(row['label_space'])
        gold = row['gold']
        require(type(gold) is int and 0 <= gold < classes, 'Invalid Q05 target')
        for scorer, prediction in row['predictions'].items():
            require(prediction is None or (type(prediction) is int and 0 <= prediction < classes),
                    'Invalid Q05 prediction')
            require(row['invalid'][scorer] == (prediction is None), 'Q05 invalid flag disagrees')
            require(row['correct'][scorer] == int(prediction == gold), 'Q05 correctness flag disagrees')
            key = tuple(row[k] for k in Q05_KEY[:-1]) + (scorer,)
            if key not in cells:
                cells[key] = dict(zip(Q05_KEY, key), class_id=None, n=0,
                    confusion_with_invalid=[[0] * (classes + 1) for _ in range(classes)],
                    invalid_count=0, ambiguous_count=0, natural_label_emission_count=0,
                    option_echo_count=0, parser_disagreement_count=0, scorer_disagreement_count=0)
            cell = cells[key]
            require(len(cell['confusion_with_invalid']) == classes, 'Q05 class count changed within a cell')
            cell['n'] += 1
            cell['confusion_with_invalid'][gold][classes if prediction is None else prediction] += 1
            cell['invalid_count'] += prediction is None
            for source, target in (
                ('ambiguous', 'ambiguous_count'),
                ('natural_label_emission', 'natural_label_emission_count'),
                ('option_echo', 'option_echo_count'),
                ('parser_disagreement', 'parser_disagreement_count'),
                ('scorer_disagreement', 'scorer_disagreement_count'),
            ):
                cell[target] += int(row[source])
    result = []
    for key in sorted(cells, key=lambda value: tuple(str(v) for v in value)):
        cell = cells[key]
        matrix = cell['confusion_with_invalid']
        cell['accuracy'] = sum(matrix[i][i] for i in range(len(matrix))) / cell['n']
        result.append(cell)
    require(bool(result), 'No Q05 records supplied')
    return result


def verify_q05_metrics(recomputed, archived):
    expected = {tuple(row[k] for k in Q05_KEY): row for row in archived if row['class_id'] is None}
    require(len(expected) == len(archived), 'Duplicate or non-panel Q05 reference cells')
    observed = {tuple(row[k] for k in Q05_KEY): row for row in recomputed}
    require(expected.keys() == observed.keys(), 'Q05 reconstructed/reference cells differ')
    for key, row in observed.items():
        for field, value in row.items():
            saved = expected[key][field]
            equal = math.isclose(value, saved, rel_tol=0, abs_tol=1e-12) if field == 'accuracy' else value == saved
            require(equal, f'Q05 reconstruction disagrees for {key}: {field}')
    return {'recomputed_cells': len(observed), 'all_fields_match': True,
            'prediction_records': sum(row['n'] for row in recomputed if row['scorer'] == 'sum'),
            'strict_invalid': sum(row['invalid_count'] for row in recomputed if row['scorer'] == 'strict'),
            'option_echoes': sum(row['option_echo_count'] for row in recomputed if row['scorer'] == 'strict'),
            'parser_disagreements': sum(row['parser_disagreement_count'] for row in recomputed if row['scorer'] == 'strict')}


def _one(rows, description):
    rows = list(rows)
    require(len(rows) == 1, f'Expected one {description}, got {len(rows)}')
    return rows[0]


def core_seed_rows(cells, seed_cells):
    """Validate absolute per-seed means against their separately saved core cells."""
    fields = ('arm', 'task', 'condition', 'k')
    lookup = {tuple(row[key] for key in fields): row for row in cells}
    require(len(lookup) == len(cells), 'Duplicate aggregate core cell')
    seen, result = set(), []
    for cell in seed_cells:
        key = tuple(cell[field] for field in fields)
        require(key not in seen and key in lookup, 'Unknown or duplicate seed core cell')
        seen.add(key)
        estimates = cell['metrics']['reference_accuracy']['paired_replicate_estimates']
        require(bool(estimates) and len({row['seed'] for row in estimates}) == len(estimates),
                'Empty or duplicated core seeds')
        average = statistics.mean(row['estimate'] for row in estimates)
        require(math.isclose(average, lookup[key]['reference_accuracy'], rel_tol=0, abs_tol=1e-12),
                'Core seed mean disagrees with saved aggregate')
        for row in estimates:
            result.append(dict(zip(fields, key), seed=row['seed'], accuracy_pct=_percent(row['estimate'])))
    require(seen == lookup.keys(), 'Core seed coverage is incomplete')
    return result


def _percent(value):
    require(isinstance(value, (float, int)) and math.isfinite(value), 'Nonfinite numeric result')
    return 100 * value


def _interval(values, reverse=False):
    require(len(values) == 2 and values[0] <= values[1], 'Invalid stored interval')
    return [_percent(-values[1]), _percent(-values[0])] if reverse else [_percent(v) for v in values]


def build_tables(data):
    """Return numerical rows; presentation formatting never feeds calculations."""
    tables = {}
    metrics = data['q05_recomputed']
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row['task'], row['arm'], row['condition'], row['k'], row['scorer'])].append(row)
    def qrows(task, arm, condition, k, scorer='sum'):
        full_arm = 'frozen' if arm == 'frozen' else task + '_' + arm
        rows = grouped[(task, full_arm, condition, k, scorer)]
        require(bool(rows), 'Missing Q05 cell')
        require(len({(r['episode_seed'], r['fit_seed']) for r in rows}) == len(rows), 'Duplicate Q05 replicate')
        return rows
    def qmetric(task, arm, condition, k):
        return statistics.mean(r['accuracy'] for r in qrows(task, arm, condition, k))
    core = data['core_cells']
    def cmetric(arm, task, condition, k):
        return _one((r for r in core if (r['arm'], r['task'], r['condition'], r['k']) ==
                     (arm, task, condition, k)), 'core cell')['reference_accuracy']
    shots = sorted({r['k'] for r in metrics})
    endpoint = max(shots)
    recipes = data['adapter_recipes']
    endpoint_rows, invalid_rows, curve_rows = [], [], []
    for arm, title, objective in Q05_ARMS:
        params = 0 if arm == 'frozen' else recipes['Q05']['shapes4_' + arm]['trainable_parameters']
        row = {'arm': title, 'objective': objective, 'parameters': params}
        invalid = {'arm': title, 'objective': objective}
        for task, _ in TASKS[:2]:
            for condition, short in [('standard', 'standard'), ('episode_remap', 'remap')]:
                row[task + '_' + short + '_pct'] = _percent(qmetric(task, arm, condition, endpoint))
                strict = qrows(task, arm, condition, endpoint, 'strict')
                invalid[task + '_' + short + '_invalid_pct'] = _percent(sum(r['invalid_count'] for r in strict)/sum(r['n'] for r in strict))
                curve_rows.append({'task': task, 'arm': title, 'objective': objective, 'condition': short,
                    **{'K' + str(k) + '_pct': _percent(qmetric(task, arm, condition, k)) for k in shots}})
        endpoint_rows.append(row)
        invalid_rows.append(invalid)
    tables['q05_endpoints'] = endpoint_rows
    tables['q05_invalid'] = invalid_rows
    tables['q05_curves'] = curve_rows
    tables['core_seed_accuracy'] = data['core_seed_accuracy']
    tables['core_transfer'] = [dict(task=task, adapter=title,
        standard_K0_pct=_percent(cmetric(arm, task, 'standard', 0)),
        standard_K16_pct=_percent(cmetric(arm, task, 'standard', 16)),
        remap_K0_pct=_percent(cmetric(arm, task, 'remap_seen_mapping', 0)),
        remap_K16_pct=_percent(cmetric(arm, task, 'remap_seen_mapping', 16)))
        for task, _ in TASKS for arm, title in CORE_ARMS]
    tables['depth'] = [dict(blocks=f'{start}--{start+3}',
        parameters=recipes['core_and_depth']['recipes'][f'window_{start}_{start+3}_r8']['trainable_parameters'],
        standard_pct=_percent(cmetric(f'depth_window_{start}_{start+3}_r8', 'shapes4', 'standard', 16)),
        remap_pct=_percent(cmetric(f'depth_window_{start}_{start+3}_r8', 'shapes4', 'remap_seen_mapping', 16)))
        for start in range(0, 28, 4)]
    tables['adapter_targets'] = [dict(study='objective', arm=arm,
        modules=len(recipe['actual_trainable_tensor_ledger']['modules']),
        parameters=recipe['trainable_parameters'], updates=recipe['completed_steps'])
        for arm, recipe in sorted(recipes['Q05'].items())]
    tables['adapter_targets'] += [dict(study='core/depth', arm=arm,
        modules=len(recipe['modules']), parameters=recipe['trainable_parameters'], updates=None)
        for arm, recipe in sorted(recipes['core_and_depth']['recipes'].items())]

    bridge = data['bridge_analysis']['paired_accuracy_relative_to_joint']
    methods = sorted(set().union(*(set(cell['methods']) for key, cell in bridge.items() if key.endswith('/16'))))
    tables['baselines'] = []
    for method in methods:
        row = {'method': method}
        for task, _ in TASKS[1:]:
            for condition, short in [('standard', 'standard'), ('episode_remap', 'remap')]:
                cell = bridge[f'{task}/{condition}/16']['methods'].get(method)
                row[task + '_' + short + '_pct'] = None if cell is None else _percent(cell['mean'])
        tables['baselines'].append(row)
    tables['baseline_uncertainty'] = []
    for task, _ in TASKS[1:]:
        for condition in ('standard', 'episode_remap'):
            row = bridge[f'{task}/{condition}/16']['methods']['late']
            tables['baseline_uncertainty'].append(dict(task=task, condition=condition,
                joint_minus_late_pp=_percent(-row['paired_difference_from_reference']),
                image_family_band_pp=_interval(row['paired_simultaneous_family_band95'], reverse=True),
                seed_t_interval_pp=_interval(row['paired_fitted_seed_interval']['t95_df2'], reverse=True)))
    tables['core_uncertainty'] = [dict(task=row['left'][1], condition=row['left'][3],
        joint_minus_late_pp=_percent(row['estimate']), seed_t_interval_pp=_interval(row['unadjusted_t95_df2']))
        for row in data['core_seed_contrasts']
        if row['left'][0] == CORE_ARMS[1][0] and row['right'][0] == CORE_ARMS[2][0]
        and row['left'][2] == 16 and row['left'][3] in ('standard', 'remap_seen_mapping')
        and row['metric'] == 'reference_accuracy']

    tables['replication'] = []
    for key, analysis in sorted(data['replication_results'].items()):
        if not key.endswith('_images'):
            continue
        for arm in ('frozen', 'connector', 'late'):
            mid = _one((m for m, value in analysis['methods'].items() if value['arm'] == arm), 'replication method')
            recipe_rows = [r for r in analysis['fitted_recipe_table'].values() if r['arm'] == arm]
            counts = {r['trainable_parameters'] for r in recipe_rows}
            require(len(counts) <= 1, 'Replication parameter count differs across seeds')
            row = {'model': key.removeprefix('D14CPU11_').removesuffix('_images'),
                   'arm': arm, 'parameters': next(iter(counts), 0)}
            for condition in ('standard', 'remap'):
                cell = _one((r for r in analysis['cells'].values() if r['method'] == mid and
                             r['condition'] == condition and r['k'] == 16), 'replication endpoint')
                row[condition + '_pct'] = _percent(cell['accuracy']['mean'])
            tables['replication'].append(row)

    tables['nonclassification'] = []
    for task, analysis in sorted(data['q11_results'].items()):
        frozen = _one((m for m, value in analysis['methods'].items() if value['arm'] == 'frozen'), 'frozen transfer method')
        adapted = _one((m for m, value in analysis['methods'].items() if value['arm'] != 'frozen'), 'adapted transfer method')
        for key, cell in sorted(analysis['cells'].items()):
            if cell['method'] != adapted:
                continue
            condition, k = cell['condition'], cell['k']
            baseline = analysis['cells'][f'{frozen}/{task}/{condition}/{k}']['accuracy']
            delta = analysis['paired_checkpoint_contrasts'][key]['adapted_minus_frozen']
            seed = analysis['paired_fitting_context_seed_contrasts'][key + '/adapted_minus_frozen']['t_interval_95']
            tables['nonclassification'].append(dict(task=task, condition=condition, k=k,
                images=cell['accuracy']['images'], frozen_pct=_percent(baseline['mean']),
                joint_pct=_percent(cell['accuracy']['mean']),
                image_interval_pp=_interval([delta['low95'], delta['high95']]), seed_t_interval_pp=_interval(seed)))

    tables['continual'] = [dict(arm=arm, streams=row['streams'], AA_pct=_percent(row['AA']), BWT_pp=_percent(row['BWT']))
        for arm, row in sorted(data['continual_results']['summary'].items())]
    tables['controller'] = [dict(policy=row['arm'], online_available_K16_pct=_percent(row['mean_online_accuracy']['16']['sum']),
        AA_pct=_percent(row['mean_retention']['sum']['AA']), BWT_pp=_percent(row['mean_retention']['sum']['BWT']),
        total_seconds=math.fsum(row['total_cost_seconds'].values()), **row['total_cost_seconds'])
        for row in data['controller_results']['policies']]
    interventions = data['intervention_results']
    tables['frozen_interventions'] = [dict(intervention=method, accuracy_pct=_percent(row['mean']),
        difference_pp=_percent(row['paired_difference_from_reference']),
        image_family_band_pp=_interval(row['paired_simultaneous_family_band95']))
        for method, row in sorted(interventions['Q09']['accuracy_vs_clean']['methods'].items())]
    tables['gate'] = []
    for model, analysis in sorted(interventions['Q15']['models'].items()):
        for family, results in sorted(analysis['accuracy_families'].items()):
            task, condition = family.split('/')
            tables['gate'].append(dict(model=model, task=task, condition=condition,
                **{method + '_pct': _percent(results['methods'][method]['mean']) for method in ('Gate', 'fixed_K0', 'fixed_K16')}))
    inject = interventions['Q16']
    tables['inject'] = [dict(task=task, method=method, accuracy_pct=_percent(row['mean']))
        for task, family in sorted(inject['accuracy_families'].items())
        for method, row in sorted(family['versus_explicit']['methods'].items())]
    tables['inject_cost'] = []
    for task in sorted(inject['accuracy_families']):
        blocks = [block for block in inject['blocks'] if block['block_id'].startswith(task + '/')]
        for method in ('explicit', 'explicit_cached_KV', 'difference'):
            costs = [_one((cost for cost in block['costs'][method]['reuse_estimates'] if cost['reuse_count'] == 80),
                          'reuse-80 cost') for block in blocks]
            tables['inject_cost'].append(dict(task=task, method=method, reuse_count=80,
                input_tokens=statistics.mean(c['input_tokens'] for c in costs),
                seconds=statistics.mean(c['seconds'] for c in costs)))
    tables['families'] = []
    for model, analysis in sorted(data['family_results'].items()):
        for cell in sorted(analysis['cells'].values(), key=lambda r: (r['condition'], r['k'])):
            tables['families'].append(dict(model=model.removesuffix(' analysis'), condition=cell['condition'], k=cell['k'],
                accuracy_pct=_percent(cell['accuracy']['mean']), strict_invalid_pct=_percent(cell['strict_invalid_rate']),
                strict_generation_accuracy_pct=_percent(cell['strict_generation_accuracy']),
                exact_format_invalid_pct=_percent(cell['exact_format_invalid_rate']),
                sum_mean_disagreement_pct=_percent(cell['sum_mean_disagreement_rate'])))
    tables['position_examples'] = [{key: row['display'][key] for key in
        ('freq_display', 'variant_display', 'query_display', 'support_display', 'donor_display', 'pred_display')}
        for row in data['position_examples']]
    contrasts = defaultdict(list)
    for family, values in data['position_results']['paired_contrasts'].items():
        _, layout, scorer = family.split('/')
        for comparison, metrics_by_name in values.items():
            for metric, value in metrics_by_name.items():
                if value is not None:
                    contrasts[(layout, scorer, comparison, metric)].append(value)
    tables['position_contrasts'] = [dict(layout=layout, scorer=scorer, comparison=comparison,
        metric=metric, difference=statistics.mean(values), replicate_count=len(values),
        units='log probability' if metric.endswith('logp') else 'proportion')
        for (layout, scorer, comparison, metric), values in sorted(contrasts.items())]
    for name, rows in tables.items():
        require(bool(rows), 'Empty result table: ' + name)
    return tables, grouped


def _formatted(value):
    if value is None:
        return '--'
    if isinstance(value, float):
        return f'{value:.2f}'
    if isinstance(value, list):
        return '[' + ', '.join(_formatted(v) for v in value) + ']'
    return str(value)


def _tex(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(mapping.get(c, c) for c in _formatted(value))


def _write_rows(output, name, rows, *, presentation=True):
    columns = list(rows[0])
    require(all(set(row) == set(columns) for row in rows), 'Nonrectangular table: ' + name)
    (output / (name + '.json')).write_text(json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with (output / (name + '.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)
    if presentation:
        lines = ['% Generated from bundled numerical evidence; requires booktabs.',
                 r'\begin{tabular}{' + 'l' * len(columns) + '}', r'\toprule',
                 ' & '.join(_tex(key) for key in columns) + r' \\', r'\midrule']
        lines += [' & '.join(_tex(row[key]) for key in columns) + r' \\' for row in rows]
        lines += [r'\bottomrule', r'\end{tabular}']
        (output / (name + '.tex')).write_text('\n'.join(lines) + '\n')
        lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
        lines += ['| ' + ' | '.join(_formatted(row[key]) for key in columns) + ' |' for row in rows]
        (output / (name + '.md')).write_text('\n'.join(lines) + '\n')


def _plots(output, data, grouped):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Figures require the analysis extra: pip install -e ".[analysis]"') from error
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    shots = sorted({row['k'] for row in data['q05_recomputed']})
    palette = ['#222222', '#0072B2', '#56B4E9', '#D55E00', '#CC79A7']
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.2), sharex=True, sharey=True)
    for ti, (task, title) in enumerate(TASKS[:2]):
        for ci, condition in enumerate(('standard', 'episode_remap')):
            ax = axes[ti, ci]
            for (arm, label, objective), color in zip(Q05_ARMS[:5], palette):
                full = 'frozen' if arm == 'frozen' else task + '_' + arm
                cells = [grouped[(task, full, condition, k, 'sum')] for k in shots]
                ys = [_percent(statistics.mean(row['accuracy'] for row in rows)) for rows in cells]
                name = label if arm == 'frozen' else label + ' / ' + objective.lower()
                ax.plot(range(len(shots)), ys, marker='o', markersize=3, linewidth=1.5, color=color, label=name)
                seeds = sorted({row['episode_seed'] for row in cells[0]})
                for seed in seeds:
                    trace = [_percent(_one((row for row in rows if row['episode_seed'] == seed), 'Q05 seed')['accuracy']) for rows in cells]
                    ax.plot(range(len(shots)), trace, color=color, alpha=.17, linewidth=.7)
            classes = len(data['q05_recomputed'][0]['confusion_with_invalid'])
            ax.axhline(100/classes, color='.65', linestyle=':', linewidth=.8)
            ax.set(title=title + ' / ' + ('Standard' if ci == 0 else 'Remap'), ylim=(0, 103))
            ax.set_xticks(range(len(shots)), shots)
            ax.grid(axis='y', alpha=.15)
            if ci == 0:
                ax.set_ylabel('Accuracy (%)')
            if ti == 1:
                ax.set_xlabel('Demonstrations K')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .105, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(output / ('q05_curves.' + suffix), dpi=180, bbox_inches='tight',
                    metadata={'CreationDate': None} if suffix == 'pdf' else None)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for ti, (task, title) in enumerate(TASKS):
        for ci, condition in enumerate(('standard', 'remap_seen_mapping')):
            ax = axes[ci, ti]
            for (arm, label), color in zip(CORE_ARMS, ('#222222', '#0072B2', '#D55E00')):
                ys = [_percent(_one((r for r in data['core_cells'] if (r['arm'], r['task'], r['condition'], r['k']) ==
                                    (arm, task, condition, k)), 'core curve cell')['reference_accuracy']) for k in shots]
                ax.plot(range(len(shots)), ys, marker='o', markersize=3, linewidth=1.5, color=color, label=label)
                seed_rows = [r for r in data['core_seed_accuracy'] if
                             (r['arm'], r['task'], r['condition']) == (arm, task, condition)]
                for seed in sorted({r['seed'] for r in seed_rows}):
                    trace = [_one((r for r in seed_rows if (r['seed'], r['k']) == (seed, k)),
                                  'core seed trace')['accuracy_pct'] for k in shots]
                    ax.plot(range(len(shots)), trace, color=color, alpha=.17, linewidth=.7)
            ax.axhline(100/classes, color='.65', linestyle=':', linewidth=.8)
            ax.set(title=title + ' / ' + ('Standard' if ci == 0 else 'Remap'), ylim=(0, 103))
            ax.set_xticks(range(len(shots)), shots)
            ax.grid(axis='y', alpha=.15)
            if ti == 0:
                ax.set_ylabel('Accuracy (%)')
            if ci == 1:
                ax.set_xlabel('Demonstrations K')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    for suffix in ('pdf', 'png'):
        fig.savefig(output / ('core_curves.' + suffix), dpi=180, bbox_inches='tight',
                    metadata={'CreationDate': None} if suffix == 'pdf' else None)
    plt.close(fig)


def regenerate(output, *, figures=True):
    """Create a fresh directory of table data, readable tables, plots and hashes.

    Probabilities become percentages only in display tables. Recomputed intervals
    retain their original image/replicate interpretation. ``figures=False`` skips
    matplotlib; statistical resampling still uses NumPy on CPU.
    """
    output = Path(output).resolve()
    require(not output.is_relative_to(source_path('data')), 'Output must not be inside the bundled evidence directory')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            'Use a fresh regeneration output directory')
    files = {
        'q05_metrics': 'q05_metrics.json.gz', 'q05_per_image': 'q05_per_image.json.gz',
        'core_cells': 'core_cells.json', 'core_seed_contrasts': 'core_seed_contrasts.json.gz',
        'core_seed_cells': 'core_seed_cells.json.gz',
        'adapter_recipes': 'adapter_recipes.json.gz', 'bridge_analysis': 'bridge_analysis.json',
        'replication_results': 'replication_results.json', 'q11_results': 'q11_results.json',
        'continual_results': 'continual_results.json', 'controller_results': 'controller_results.json',
        'intervention_results': 'intervention_results.json.gz', 'family_results': 'family_results.json',
        'position_results': 'position_results.json', 'position_examples': 'position_examples.json',
    }
    data, inputs = {}, {}
    for key, name in files.items():
        path = source_path('data/paper/' + name)
        data[key] = read_json(path)
        inputs['data/paper/' + name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    from .q05_analysis import reconstruct_q05, verify_q05_outcomes
    from .position_analysis import analyze_position, verify_position
    from .native_analysis import recompute_native
    from .analysis import analyze
    output.mkdir(parents=True, exist_ok=True)
    analysis_dir = output / 'analysis'
    data.update(analyze(analysis_dir))
    native = recompute_native()
    data.update({key: native[key] for key in ('family_results', 'replication_results')})
    for key in ('native_summary', 'native_validation'):
        (analysis_dir / (key + '.json')).write_text(json.dumps(native[key], indent=2, sort_keys=True, allow_nan=False)+'\n')
    position = analyze_position()
    position_validation = verify_position(position, data['position_results'])
    data['position_results'].update(position)
    (analysis_dir / 'position_results.json').write_text(json.dumps(position, indent=2, sort_keys=True, allow_nan=False)+'\n')
    q05_rows = reconstruct_q05()
    q05_outcome_validation = verify_q05_outcomes(q05_rows, data.pop('q05_per_image'))
    data['q05_recomputed'] = aggregate_q05(q05_rows)
    validation = verify_q05_metrics(data['q05_recomputed'], data['q05_metrics'])
    validation.update(q05_outcome_validation)
    data['core_seed_accuracy'] = core_seed_rows(data['core_cells'], data['core_seed_cells'])
    tables, grouped = build_tables(data)
    tables_dir, figures_dir = output / 'tables', output / 'figures'
    tables_dir.mkdir(parents=True)
    for name, rows in tables.items():
        _write_rows(tables_dir, name, rows, presentation=name != 'core_seed_accuracy')
    _write_rows(tables_dir, 'q05_per_cell_diagnostics', data['q05_recomputed'], presentation=False)
    _write_rows(tables_dir, 'core_all_cells', data['core_cells'], presentation=False)
    if figures:
        figures_dir.mkdir()
        _plots(figures_dir, data, grouped)
    scope = ('Q05, native-family/replication and position predictions are recalculated from original '
             'generated answers and complete candidate token scores. Core, baseline, open-task, '
             'intervention and sequential analyses are rebuilt from released per-example records, '
             'including image resampling with the recorded RNG/settings and calibration refits. '
             'Archived summaries are checked as references; no model inference or fitting occurs. Repeated '
             'contexts are not independent query images. Candidate scoring and strict generation '
             'are separate. All intervals are descriptive for their recorded fixed panels. '
             'Both plots show retained cell means and recorded paired fitting/context seed traces. '
             'Tables reproduce numerical evidence in portable layouts, not journal pagination.')
    for path in sorted(source_path('data/analysis').rglob('*')):
        if path.is_file():
            inputs['data/analysis/' + str(path.relative_to(source_path('data/analysis')))] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size}
    manifest = {'schema_version': 2, 'scope': scope, 'inputs': inputs,
                'q05_validation': validation,
                'native_validation': native['native_validation'],
                'position_validation': position_validation,
                'statistical_analysis_validation': data.get('analysis_validation'),
                'core_seed_validation': {'aggregate_cells': len(data['core_cells']),
                                         'seed_accuracy_rows': len(data['core_seed_accuracy']),
                                         'all_means_match': True},
                'tables': {name: len(rows) for name, rows in tables.items()},
                'figures_generated': bool(figures),
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    import numpy
    manifest['analysis_runtime'] = {'python': sys.version, 'numpy': numpy.__version__}
    manifest['analysis_code_sha256'] = {
        str(path.relative_to(source_path('vlm_context'))): sha256(path)
        for path in sorted(source_path('vlm_context').glob('*.py'))
        if 'analysis' in path.name or path.name in ('regenerate.py', 'protocol.py', 'native_families.py', 'open_tasks.py', 'io.py')}
    manifest['outputs'] = {str(path.relative_to(output)): {'sha256': sha256(path), 'bytes': path.stat().st_size}
                           for path in sorted(output.rglob('*')) if path.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    (output / 'README.txt').write_text(scope + '\n\nCSV/JSON retain unrounded numeric values. '
        'Markdown/LaTeX tables round for reading; suffix _pct denotes percent and _pp percentage points. '
        'LaTeX fragments require booktabs. K16 on controller tables means the available prefix, '
        'not the prefix selected by the policy. Native VQA scores and paired-rule joint success '
        'must not be interpreted as ordinary single-answer classification accuracy.\n')
    return manifest
