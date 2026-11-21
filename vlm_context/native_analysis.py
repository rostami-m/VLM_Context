"""CPU reduction of all released native-family and replication model outputs.

No model is loaded. Candidate probabilities, generated answers, query identities
and paired contexts are sufficient to reconstruct the reported score results and
both normalized-label and case-sensitive formatting diagnostics. Image intervals
reproduce the original unstratified paired image bootstrap (seed 20261106);
contexts are averaged within each image, never treated as independent images.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import random
import statistics

from .io import read_json, require, source_path, write_json, sha256
from .native_families import exact_format_parse, strict_parse


def bootstrap(values, *, draws=10000, seed=20261106):
    """Preserve the archived image order and explicit percentile convention."""
    require(values and draws >= 100, 'Bootstrap needs observations and at least 100 draws')
    require(all(math.isfinite(v) for v in values), 'Nonfinite bootstrap observation')
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(rng.choice(values) for _ in values) / n for _ in range(draws))
    return dict(mean=sum(values) / n, low95=samples[int(.025 * draws)],
                high95=samples[int(.975 * draws)], images=n, draws=draws,
                sampling_unit='query image, seeds averaged within image')


def image_summary(outcomes, *, draws=10000):
    images = defaultdict(dict)
    for (image, seed), value in outcomes.items():
        images[image][seed] = value
    require(images and all(set(seeds) == {0, 1, 2} for seeds in images.values()),
            'Every native query must have all three paired contexts')
    return bootstrap([sum(seeds.values()) / 3 for seeds in images.values()], draws=draws)


def seed_interval(left, right):
    require(left.keys() == right.keys(), 'Unpaired native seed contrast')
    by_seed = defaultdict(list)
    for (image, seed), value in left.items():
        by_seed[seed].append(value - right[image, seed])
    require(set(by_seed) == {0, 1, 2}, 'Three native fitting/context replicates required')
    estimates = {str(seed): math.fsum(by_seed[seed]) / len(by_seed[seed]) for seed in (0, 1, 2)}
    values = list(estimates.values())
    mean, sd = statistics.mean(values), statistics.stdev(values)
    critical = math.sqrt(2 * .95**2 / (1 - .95**2))
    half = critical * sd / math.sqrt(3)
    return dict(paired_replicate_estimates=estimates, replicates=3, mean=mean,
                sample_sd=sd, t_degrees_of_freedom=2, t_critical_975=critical,
                t_interval_95=[mean - half, mean + half], range=[min(values), max(values)],
                interval_not_clipped_to_parameter_range=True,
                unique_query_images=len({image for image, _ in left}),
                conditioning='fixed query panel; paired fitting and context replicate varies jointly')


def score_record(row):
    """Check complete-answer score arithmetic before deriving any predictions."""
    labels, options = row['label_space'], row['displayed_options']
    require(len(labels) == len(set(labels)) and len(labels) > 1,
            'Empty/duplicate native label inventory')
    require(len(options) == len(labels) and set(options) == set(labels),
            'Native displayed options differ from labels')
    require(type(row['gold']) is int and 0 <= row['gold'] < len(labels), 'Invalid native gold index')
    require(type(row['seed']) is int and row['seed'] in (0, 1, 2), 'Invalid native context seed')
    candidates = row['candidates']
    require([c['label'] for c in candidates] == options, 'Native candidate order differs')
    for candidate in candidates:
        tokens, label_tokens = candidate['scored_token_ids'], candidate['label_token_ids']
        end = candidate['native_end_token']
        require(label_tokens and tokens == label_tokens + [end] and end not in label_tokens,
                'Candidate is not a complete answer plus one native end token')
        require(all(type(token) is int and token >= 0 for token in tokens), 'Invalid candidate token')
        probabilities = candidate['token_log_probabilities']
        require(len(probabilities) == len(tokens) and
                all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in probabilities),
                'Invalid candidate log probabilities')
        for actual, expected in [(sum(probabilities), candidate['sum_logp']),
                                 (sum(probabilities) / len(probabilities), candidate['mean_logp'])]:
            require(math.isfinite(expected) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6),
                    'Candidate score arithmetic differs')
    summed = max(candidates, key=lambda c: sum(c['token_log_probabilities']))['label']
    mean = max(candidates, key=lambda c: sum(c['token_log_probabilities']) / len(c['scored_token_ids']))['label']
    require(summed == row['recorded_sum_prediction'] and mean == row['recorded_mean_prediction'],
            'Recomputed candidate argmax differs from recorded prediction')
    require(isinstance(row['greedy_text'], str), 'Missing native generation')
    strict = strict_parse(row['greedy_text'], options)
    exact = exact_format_parse(row['greedy_text'], options)
    require(exact == row['recorded_exact_format_prediction_index'],
            'Archived exact-format prediction differs from generated answer')
    return dict(sum=labels.index(summed), mean=labels.index(mean),
                strict=None if strict is None else labels.index(options[strict]),
                exact_format=None if exact is None else labels.index(options[exact]),
                sum_position=options.index(summed))


def reduce_records(records, methods, *, draws=10000):
    cells, outcomes, seen = {}, defaultdict(dict), set()
    for row in records:
        method, task, condition, k = (row[f] for f in ('method', 'task', 'condition', 'k'))
        require(method in methods, 'Unknown native method')
        if 'checkpoint_descriptors' in methods[method]:
            expected_checkpoint = methods[method]['checkpoint_descriptors'].get(str(row['seed']))
            require(row['checkpoint_descriptor_sha256'] == expected_checkpoint,
                    'Native record checkpoint differs from its declared fitting seed')
        key = (method, task, condition, k)
        identity = (*key, row['image_id'], row['seed'])
        require(identity not in seen, 'Duplicate native image/context/checkpoint record')
        seen.add(identity)
        predictions = score_record(row)
        gold, nclasses = row['gold'], len(row['label_space'])
        if key not in cells:
            cells[key] = dict(method=method, task=task, condition=condition, k=k, records=0,
                              strict_confusion_with_invalid_column=[[0]*(nclasses+1) for _ in range(nclasses)],
                              exact_format_confusion_with_invalid_column=[[0]*(nclasses+1) for _ in range(nclasses)],
                              sum_confusion=[[0]*nclasses for _ in range(nclasses)],
                              mean_confusion=[[0]*nclasses for _ in range(nclasses)],
                              sum_option_position_counts=[0]*nclasses, counts=Counter())
        cell = cells[key]
        require(len(cell['sum_confusion']) == nclasses, 'Native label count changes within cell')
        cell['records'] += 1
        for scorer in ('sum', 'mean', 'strict', 'exact_format'):
            prediction = predictions[scorer]
            matrix = cell[scorer + ('_confusion_with_invalid_column' if scorer in ('strict', 'exact_format') else '_confusion')]
            matrix[gold][nclasses if prediction is None else prediction] += 1
            cell['counts'][scorer + '_correct'] += prediction == gold
            cell['counts'][scorer + '_invalid'] += prediction is None
        cell['counts']['normalized_format_recoveries'] += predictions['strict'] is not None and predictions['exact_format'] is None
        cell['counts']['sum_mean_disagreement'] += predictions['sum'] != predictions['mean']
        cell['sum_option_position_counts'][predictions['sum_position']] += 1
        outcomes[key][row['image_id'], row['seed']] = float(predictions['sum'] == gold)
    require(cells, 'No native outputs supplied')
    summaries, kcontrasts, armcontrasts, seedcontrasts = {}, {}, {}, {}
    for key, cell in sorted(cells.items()):
        method, task, condition, k = key
        name = '/'.join(map(str, key))
        n, counts = cell['records'], cell.pop('counts')
        cell.update(accuracy=image_summary(outcomes[key], draws=draws),
                    strict_invalid_rate=counts['strict_invalid']/n,
                    strict_generation_accuracy=counts['strict_correct']/n,
                    exact_format_invalid_rate=counts['exact_format_invalid']/n,
                    exact_format_generation_accuracy=counts['exact_format_correct']/n,
                    normalized_format_recoveries=counts['normalized_format_recoveries'],
                    mean_score_accuracy=counts['mean_correct']/n,
                    sum_mean_disagreement_rate=counts['sum_mean_disagreement']/n)
        summaries[name] = cell
        zero = (method, task, condition, 0)
        if k and zero in outcomes:
            require(outcomes[key].keys() == outcomes[zero].keys(), 'Unpaired native K contrast')
            delta = {identity: outcomes[zero][identity] - value for identity, value in outcomes[key].items()}
            kcontrasts[name] = {'paired_K0_minus_K': image_summary(delta, draws=draws),
                                'prespecified_drop_margin': .10}
        if methods[method]['arm'] != 'frozen':
            references = [m for m, meta in methods.items() if meta['arm'] == 'frozen' and
                          meta['model_id'] == methods[method]['model_id']]
            require(len(references) == 1, 'Missing or ambiguous native frozen reference')
            baseline = outcomes[references[0], task, condition, k]
            require(outcomes[key].keys() == baseline.keys(), 'Unpaired native checkpoint contrast')
            delta = {identity: value - baseline[identity] for identity, value in outcomes[key].items()}
            armcontrasts[name] = {'adapted_minus_frozen': image_summary(delta, draws=draws),
                                  'frozen_method': references[0], 'same_image_seed_pairs': True,
                                  'fixed_deterministic_K0_baseline_reused_across_fit_seeds': False}
            seedcontrasts[name + '/adapted_minus_frozen'] = seed_interval(outcomes[key], baseline)
            if methods[method]['arm'] == 'connector':
                late = [m for m, meta in methods.items() if meta['arm'] == 'late' and
                        meta['model_id'] == methods[method]['model_id']]
                require(len(late) == 1, 'Native connector contrast requires one late reference')
                seedcontrasts[name + '/arm_minus_late'] = seed_interval(outcomes[key], outcomes[late[0], task, condition, k])
            if k:
                seedcontrasts[name + '/K_minus_K0'] = seed_interval(outcomes[key], outcomes[zero])
    return dict(records=len(seen), cells=summaries, paired_K_contrasts=kcontrasts,
                paired_checkpoint_contrasts=armcontrasts, paired_seed_contrasts=seedcontrasts)


DIAGNOSTIC_FIELDS = ('strict_confusion_with_invalid_column', 'strict_invalid_rate',
                     'strict_generation_accuracy', 'exact_format_confusion_with_invalid_column',
                     'exact_format_invalid_rate', 'exact_format_generation_accuracy',
                     'normalized_format_recoveries', 'sum_confusion', 'mean_confusion', 'mean_score_accuracy')


def _close(actual, expected, description):
    if isinstance(expected, dict):
        for key, value in expected.items():
            _close(actual[key], value, description + '/' + str(key))
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Native length differs: ' + description)
        for i, (a, e) in enumerate(zip(actual, expected)):
            _close(a, e, description + '/' + str(i))
    elif isinstance(expected, (int, float)):
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), 'Native numeric result differs: ' + description)
    else:
        require(actual == expected, 'Native metadata differs: ' + description)


def reconcile(reduced, family_results, replication_results):
    """Verify original likelihood analyses, then attach the shared diagnostics.

    The saved reference provides metadata and an independent regression target;
    every returned accuracy, image interval and fitting/context seed contrast is
    rebuilt from raw records. No saved interval serves as an analysis input.
    """
    family_results, replication_results = deepcopy(family_results), deepcopy(replication_results)
    checked = set()
    for destination in (family_results, replication_results):
        for analysis_key, analysis in destination.items():
            for name, original in analysis.get('cells', {}).items():
                recalculated = reduced['cells'][name]
                for field in ('accuracy', 'records', 'sum_mean_disagreement_rate', 'sum_option_position_counts'):
                    _close(recalculated[field], original[field], name + '/' + field)
                original['accuracy'] = recalculated['accuracy']
                for field in DIAGNOSTIC_FIELDS:
                    original[field] = recalculated[field]
                checked.add(name)
            for name, value in analysis.get('paired_K_contrasts', {}).items():
                _close(reduced['paired_K_contrasts'][name]['paired_K0_minus_K'], value['paired_K0_minus_K'], name)
                value['paired_K0_minus_K'] = reduced['paired_K_contrasts'][name]['paired_K0_minus_K']
            for name, value in analysis.get('paired_checkpoint_contrasts', {}).items():
                _close(reduced['paired_checkpoint_contrasts'][name]['adapted_minus_frozen'], value['adapted_minus_frozen'], name)
                value['adapted_minus_frozen'] = reduced['paired_checkpoint_contrasts'][name]['adapted_minus_frozen']
            for name, value in analysis.get('contrasts', {}).items():
                rebuilt = reduced['paired_seed_contrasts'][name]
                for field in rebuilt:
                    _close(rebuilt[field], value[field], name + '/' + field)
                    value[field] = rebuilt[field]
            if analysis.get('cells'):
                analysis['generation_diagnostic_protocol'] = {
                    'strict': 'Shared normalized whole-label grammar: casefold, whitespace normalization, optional trailing .!?; no prose or multiple labels.',
                    'exact_format': 'Separate case/punctuation-sensitive whole-label compliance after stripping outer whitespace.',
                    'source': 'data/analysis/native_records.json.gz',
                    'reducer': 'vlm_context.native_analysis',
                }
    require(checked == reduced['cells'].keys(), 'Native release does not cover exactly the published cells')
    return dict(family_results=family_results, replication_results=replication_results,
                native_summary=reduced,
                native_validation=dict(records=reduced['records'], cells=len(checked),
                                       likelihood_and_archived_intervals_match=True,
                                       generation_parser='shared normalized whole-label grammar',
                                       exact_format_reported_separately=True))


def recompute_native(*, draws=10000):
    manifest = read_json(source_path('data/analysis/native_manifest.json'))
    path = source_path(manifest['records_path'])
    require(sha256(path) == manifest['compressed_sha256'], 'Native compressed data hash differs')
    require(hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == manifest['decompressed_sha256'],
            'Native decompressed data hash differs')
    rows = read_json(path)
    require(len(rows) == manifest['records'], 'Native record inventory incomplete')
    reduced = reduce_records(rows, manifest['methods'], draws=draws)
    return reconcile(reduced, read_json(source_path('data/paper/family_results.json')),
                     read_json(source_path('data/paper/replication_results.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = recompute_native()
    for name, value in result.items():
        write_json(args.output / (name + '.json'), value)
    print(json.dumps(result['native_validation'], sort_keys=True))


if __name__ == '__main__':
    main()
