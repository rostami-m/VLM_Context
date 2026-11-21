"""Rebuild continual/controller results from released per-image model outputs.

The continual bootstrap preserves the original 400 image clusters and 20
task/class strata, averaging the nine stream outcomes within each image first.
Controller summaries reuse these same images across policies and prefixes;
they are descriptive means with no invented independence or confidence interval.
Recorded time components are summed, not estimated from new CPU wall time.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
import random
import statistics

from .io import read_json, require, source_path, write_json
from .protocol import parse_diagnostics, sequence_likelihoods

SCORES = ('sum', 'mean', 'strict', 'legacy')
ARMS = ('joint', 'late', 'all_depth', 'late_replay')
POLICIES = ('prompt_only_gate', 'fixed_schedule_consolidation', 'gate_triggered_consolidation')
PREFIXES = (0, 1, 2, 4, 8, 16)
COSTS = ('probe_seconds', 'prediction_seconds', 'training_seconds', 'reload_seconds')
COMPARISONS = ((POLICIES[1], POLICIES[0]), (POLICIES[2], POLICIES[0]), (POLICIES[2], POLICIES[1]))


def predictions(row):
    require(row['labels'] == ['A', 'B', 'C', 'D'], 'Stream label inventory changed')
    require(type(row['gold']) is int and row['gold'] in range(4), 'Invalid stream gold label')
    tokens, probabilities = row['candidate_token_ids'], row['token_logps']
    require(len(tokens) == len(probabilities) == 4, 'Incomplete stream candidates')
    endings = set()
    for ids, values in zip(tokens, probabilities):
        require(len(ids) == len(values) >= 2 and all(type(i) is int and i >= 0 for i in ids),
                'Incomplete stream answer/end-token evidence')
        require(ids[-1] not in ids[:-1], 'Stream answer contains an internal end token')
        endings.add(ids[-1])
    require(len(endings) == 1, 'Stream candidates disagree about native end token')
    scores = sequence_likelihoods(probabilities)
    parser = parse_diagnostics(row['greedy_answer'], row['labels'])
    return dict(sum=scores['sum_prediction'], mean=scores['mean_prediction'],
                strict=parser['prediction'], legacy=parser['legacy_prediction'])


def collect(records, metadata, *, controller=False):
    """Validate paired matrices and reduce every scorer from candidate outputs."""
    streams = {}
    for stream in metadata:
        key = stream['arm'], stream['stream_id']
        require(key not in streams, 'Duplicate stream metadata')
        require(sorted(stream['task_order']) == list(range(5)), 'Invalid five-task order')
        streams[key] = dict(meta=stream, matrix=defaultdict(dict), online=defaultdict(dict))
    seen, query_labels = set(), {}
    for row in records:
        key = row['arm'], row['stream_id']
        require(key in streams, 'Raw output has no stream metadata')
        stream = streams[key]
        require(row['task_order'] == stream['meta']['task_order'], 'Raw stream task order differs')
        require(type(row['stage']) is int and row['stage'] in range(5), 'Invalid stream stage')
        task, stage = row['task'], row['stage']
        require(task in row['task_order'][:stage+1], 'Evaluation uses a future task')
        if row['kind'] == 'matrix':
            require(row['k'] == 0, 'Retention must be zero-shot')
            group, group_key = stream['matrix'], (stage, task)
        else:
            require(controller and row['kind'] == 'online', 'Unexpected stream outcome kind')
            require(task == row['task_order'][stage], 'Online query uses a different task')
            available = row['available_prefix']
            require(available in PREFIXES and type(row['k']) is int and 0 <= row['k'] <= available,
                    'Selected context exceeds available prefix')
            group, group_key = stream['online'], (stage, available)
        identity = (*key, row['kind'], *group_key, row['image_id'])
        require(identity not in seen, 'Duplicate stream query output')
        seen.add(identity)
        label_identity = task, row['gold']
        require(query_labels.setdefault(row['image_id'], label_identity) == label_identity,
                'Query image changes task or class')
        predicted = predictions(row)
        group[group_key][row['image_id']] = dict(gold=row['gold'], task=task, k=row['k'],
                                                 correct={score: float(predicted[score] == row['gold']) for score in SCORES})
    cohorts = {}
    for stream in streams.values():
        order = stream['meta']['task_order']
        expected = {(stage, task) for stage in range(5) for task in order[:stage+1]}
        require(set(stream['matrix']) == expected, 'Incomplete retention matrix')
        matrices = {score: [[None]*5 for _ in range(5)] for score in SCORES}
        for (stage, task), rows in stream['matrix'].items():
            require(len(rows) == 80, 'Every retention task needs exactly 80 query images')
            require(all(sum(row['gold'] == cls for row in rows.values()) == 20 for cls in range(4)),
                    'Retention task/class strata changed')
            require(cohorts.setdefault(task, set(rows)) == set(rows), 'Stream query cohorts are not paired')
            col = order.index(task)
            for score in SCORES:
                matrices[score][stage][col] = sum(r['correct'][score] for r in rows.values()) / 80
        stream['matrices'] = matrices
        stream['retention'] = {score: dict(AA=sum(matrix[-1])/5,
                                          BWT=sum(matrix[-1][j]-matrix[j][j] for j in range(4))/4)
                               for score, matrix in matrices.items()}
        if controller:
            require(set(stream['online']) == {(stage, k) for stage in range(5) for k in PREFIXES},
                    'Incomplete online policy prefix grid')
            for (stage, _), rows in stream['online'].items():
                require(set(rows) == cohorts[order[stage]], 'Online and retention image cohorts differ')
                require(len({row['k'] for row in rows.values()}) == 1, 'One online decision selects different contexts')
    require(len(query_labels) == 400 and len(cohorts) == 5, 'Expected the same 400 images across streams')
    return streams


def bootstrap(values, *, seed=0, draws=2000):
    """Original stratified paired-image estimator and percentile indices."""
    strata = defaultdict(list)
    for task, cls, value in values.values():
        strata[task, cls].append(value)
    require(len(values) == 400 and len(strata) == 20 and all(len(group) == 20 for group in strata.values()),
            'Bootstrap requires 400 images in 20 task/class strata')
    require(draws >= 100, 'At least 100 bootstrap draws required')
    rng = random.Random(seed)
    groups = [strata[key] for key in sorted(strata)]
    samples = sorted(sum(group[rng.randrange(20)] for group in groups for _ in range(20)) / 400
                     for _ in range(draws))
    return dict(estimate=statistics.mean(v[2] for v in values.values()),
                descriptive_95_interval=[samples[int(.025*(draws-1))], samples[int(.975*(draws-1))]],
                unique_images=400, strata=20, draws=draws)


def reduce_continual(records, metadata, *, draws=2000):
    streams = collect(records, metadata)
    expected = {(arm, 'order_' + str(order) + '_seed_' + str(seed))
                for arm in ARMS for order in range(3) for seed in range(3)}
    require(set(streams) == expected, 'All 36 continual streams are required')
    metrics, image_values, matrices = [], defaultdict(lambda: defaultdict(list)), []
    for (arm, stream_id), stream in sorted(streams.items()):
        order = stream['meta']['task_order']
        matrices.append(dict(arm=arm, stream_id=stream_id, task_order=order, matrices=stream['matrices']))
        for score in SCORES:
            metrics.append(dict(arm=arm, stream=stream_id, score=score, **stream['retention'][score]))
        for task in order:
            learned = order.index(task)
            final, diagonal = stream['matrix'][4, task], stream['matrix'][learned, task]
            for image, row in final.items():
                value = row['correct']['sum']
                image_values[arm, 'AA'][image].append((task, row['gold'], value))
                delta = (value - diagonal[image]['correct']['sum']) * 5/4 if learned < 4 else 0.
                image_values[arm, 'BWT'][image].append((task, row['gold'], delta))
    collapsed = {}
    for key, images in image_values.items():
        require(len(images) == 400 and all(len(values) == 9 for values in images.values()),
                'Seeds/orders cannot become additional independent image clusters')
        collapsed[key] = {image: (values[0][0], values[0][1], statistics.mean(v[2] for v in values))
                          for image, values in images.items()}
        require(math.isclose(statistics.mean(v[2] for v in collapsed[key].values()),
                             statistics.mean(r[key[1]] for r in metrics if r['arm'] == key[0] and r['score'] == 'sum'),
                             rel_tol=0, abs_tol=1e-12), 'Image-level and stream-level continual means differ')
    contrasts = []
    for arm in ('joint', 'all_depth', 'late_replay'):
        for metric in ('AA', 'BWT'):
            a, b = collapsed[arm, metric], collapsed['late', metric]
            require(a.keys() == b.keys(), 'Unpaired continual contrast')
            values = {image: (a[image][0], a[image][1], a[image][2]-b[image][2]) for image in a}
            contrasts.append(dict(arm=arm, reference='late', metric=metric,
                                  **bootstrap(values, seed=len(contrasts), draws=draws)))
    summary = {arm: dict(streams=9,
                         **{metric: statistics.mean(r[metric] for r in metrics if r['arm'] == arm and r['score'] == 'sum')
                            for metric in ('AA', 'BWT')}) for arm in ARMS}
    return dict(streams=metrics, summary=summary, paired_primary_contrasts=contrasts), matrices


def _controller_events(stream):
    events = stream['meta']['events']
    require(len(events) == 30, 'Five stages and six controller prefix decisions required')
    expected = {(stage, prefix) for stage in range(5) for prefix in PREFIXES}
    require({(event['stage_index'], event['available_prefix']) for event in events} == expected,
            'Controller event grid is incomplete or duplicated')
    steps = consolidations = charged = reused = 0
    for event in events:
        stage, prefix = event['stage_index'], event['available_prefix']
        require(event['task'] == stream['meta']['task_order'][stage], 'Controller event task order differs')
        selected = {row['k'] for row in stream['online'][stage, prefix].values()}
        gate = event['gate_after'] if event['consolidation'] else event['gate_before']
        require(selected == {gate['selected_k']}, 'Online outputs disagree with controller selected context')
        for name in ('gate_before', 'gate_after'):
            if name in event:
                gate = event[name]
                require(all(type(gate[field]) is int and gate[field] >= 0
                            for field in ('charged_queries', 'cached_queries_reused')), 'Invalid controller probe count')
                charged += gate['charged_queries']
                reused += gate['cached_queries_reused']
        if event['consolidation']:
            require(event['training_steps'] == 80, 'Controller consolidation budget differs')
            steps += 80
            consolidations += 1
    return dict(optimizer_steps=steps, consolidation_events=consolidations,
                charged_probe_queries=charged, cached_probe_queries_reused=reused)


def reduce_controller(records, metadata):
    streams = collect(records, metadata, controller=True)
    expected = {(arm, 'order_0_seed_' + str(seed)) for arm in POLICIES for seed in range(3)}
    require(set(streams) == expected, 'All nine reported controller streams are required')
    rows, matrices = [], []
    for meta in metadata:
        arm, stream_id = meta['arm'], meta['stream_id']
        stream = streams[arm, stream_id]
        costs = meta['costs']
        require(set(costs) == set(COSTS) and all(math.isfinite(v) and v >= 0 for v in costs.values()),
                'Invalid recorded method timing components')
        online = {}
        for available in PREFIXES:
            selected = [stream['online'][stage, available] for stage in range(5)]
            correct = {score: int(sum(row['correct'][score] for group in selected for row in group.values()))
                       for score in SCORES}
            online[str(available)] = dict(correct=correct, queries=400,
                                           accuracy={score: correct[score]/400 for score in SCORES},
                                           mean_selected_k=sum(next(iter(group.values()))['k'] for group in selected)/5)
        rows.append(dict(arm=arm, stream_id=stream_id, task_order=meta['task_order'],
                         initialization_seed=int(stream_id.rsplit('_', 1)[1]), online=online,
                         retention=stream['retention'], costs=costs, **_controller_events(stream)))
        matrices.append(dict(arm=arm, stream_id=stream_id, task_order=meta['task_order'], matrices=stream['matrices']))
    policies = []
    for arm in POLICIES:
        subset = [row for row in rows if row['arm'] == arm]
        policies.append(dict(arm=arm, streams=3, unique_final_images=400,
                             mean_online_accuracy={str(k): {score: sum(row['online'][str(k)]['accuracy'][score] for row in subset)/3
                                                            for score in SCORES} for k in PREFIXES},
                             mean_retention={score: {metric: sum(row['retention'][score][metric] for row in subset)/3
                                                     for metric in ('AA', 'BWT')} for score in SCORES},
                             total_cost_seconds={name: sum(row['costs'][name] for row in subset) for name in COSTS}))
    index = {(row['arm'], row['stream_id']): row for row in rows}
    contrasts = []
    for treatment, reference in COMPARISONS:
        paired = []
        for stream_id in ('order_0_seed_0', 'order_0_seed_1', 'order_0_seed_2'):
            a, b = index[treatment, stream_id], index[reference, stream_id]
            paired.append(dict(stream_id=stream_id,
                               online_accuracy_difference={str(k): {score: a['online'][str(k)]['accuracy'][score]-b['online'][str(k)]['accuracy'][score]
                                                                      for score in SCORES} for k in PREFIXES},
                               retention_difference={score: {metric: a['retention'][score][metric]-b['retention'][score][metric]
                                                               for metric in ('AA', 'BWT')} for score in SCORES},
                               cost_seconds_difference={name: a['costs'][name]-b['costs'][name] for name in COSTS}))
        contrasts.append(dict(treatment=treatment, reference=reference,
                              direction='treatment minus reference', paired_seeds=paired))
    return dict(streams=rows, policies=policies, paired_policy_contrasts=contrasts,
                task_orders=1, seeds=3), matrices


def verify(actual, expected, prefix=''):
    if isinstance(expected, dict):
        for key, value in expected.items():
            verify(actual[key], value, prefix + '/' + str(key))
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Stream result length differs: ' + prefix)
        for i, (a, e) in enumerate(zip(actual, expected)):
            verify(a, e, prefix + '/' + str(i))
    elif isinstance(expected, (int, float)):
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), 'Stream result differs: ' + prefix)
    else:
        require(actual == expected, 'Stream metadata differs: ' + prefix)


def recompute_continual_controller():
    continual_rows = read_json(source_path('data/analysis/continual_raw_scores.json.gz'))
    controller_rows = read_json(source_path('data/analysis/controller_raw_scores.json.gz'))
    continual, continual_matrices = reduce_continual(continual_rows, read_json(source_path('data/analysis/continual_streams.json.gz')))
    controller, controller_matrices = reduce_controller(controller_rows, read_json(source_path('data/analysis/controller_streams.json.gz')))
    old_continual = read_json(source_path('data/paper/continual_results.json'))
    old_controller = read_json(source_path('data/paper/controller_results.json'))
    # Only narrative scope is retained; every numerical result above comes from raw inputs.
    continual['scope'] = old_continual['scope']
    controller['limits'] = old_controller['limits']
    verify(continual, old_continual)
    verify(controller, old_controller)
    return dict(continual_results=continual, controller_results=controller,
                stream_matrices=dict(continual=continual_matrices, controller=controller_matrices),
                stream_validation=dict(continual_records=len(continual_rows), controller_records=len(controller_rows),
                                       continual_streams=36, controller_streams=9,
                                       continual_paired_image_intervals=6, unique_query_images=400,
                                       all_archived_numerical_results_match=True,
                                       timed_work='Recorded separate timing components; no new GPU execution.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = recompute_continual_controller()
    for name, value in results.items():
        write_json(args.output / (name + '.json'), value)
    print(json.dumps(results['stream_validation'], sort_keys=True))


if __name__ == '__main__':
    main()
