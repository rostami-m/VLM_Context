"""Rebuild the intact-pair position controls from saved answers and token scores."""
from collections import defaultdict
import math
import statistics

from .io import read_json, require, source_path
from .protocol import parse_diagnostics, sequence_likelihoods


def analyze_position(rows=None):
    rows = read_json(source_path('data/analysis/position_raw.json.gz')) if rows is None else rows
    cells, seen = defaultdict(dict), set()
    for row in rows:
        identity = (row['seed'], row['layout'], row['query_id'], row['variant'])
        require(identity not in seen, 'Duplicate position outcome')
        seen.add(identity)
        require(len(row['support_classes']) == 8, 'Position control requires eight supports')
        require(len(row['label_space']) == len(row['candidate_tokens']) == len(row['per_token_label_log_probabilities']),
                'Position candidate count differs')
        for label, token, values in zip(row['label_space'], row['candidate_tokens'], row['per_token_label_log_probabilities']):
            require(token['label'] == label and token['scored_token_ids'] == token['label_token_ids'] + [row['eos_token_id']],
                    'Position candidate/end-token convention differs')
            require(len(token['scored_token_ids']) == len(values), 'Position token alignment differs')
        scores = sequence_likelihoods(row['per_token_label_log_probabilities'])
        predictions = dict(sum=scores['sum_prediction'], mean=scores['mean_prediction'],
                           strict=parse_diagnostics(row['greedy_answer'], row['label_space'])['prediction'])
        for scorer, prediction in predictions.items():
            metrics = dict(accuracy=float(prediction == row['gold']),
                any_support=float(prediction in row['support_classes']),
                correct_label_logp=scores['sum'][row['gold']],
                first=float(prediction == row['support_classes'][0]),
                last=float(prediction == row['support_classes'][-1]),
                majority_set=float(prediction in row['majority_classes']),
                unique_majority=(float(prediction == row['majority_classes'][0])
                                 if len(row['majority_classes']) == 1 else None),
                wrong_donor_class=float(prediction == row['wrong_class']))
            key = f"seed{row['seed']}/{row['layout']}/{scorer}/{row['variant']}"
            cells[key][row['query_id']] = metrics
    output = {'raw_records': len(rows), 'cells': {}, 'paired_contrasts': {}}
    for key, values in cells.items():
        require(len(values) == 80, 'Position cell must contain 80 distinct queries')
        first = next(iter(values.values()))
        output['cells'][key] = dict(n=len(values), metrics={name:
            None if first[name] is None else statistics.mean(row[name] for row in values.values()) for name in first})
    for prefix in sorted({key.rsplit('/', 1)[0] for key in cells}):
        contrasts = {}
        for left, right in [('interior_sham', 'baseline'), ('wrong_endpoint', 'baseline'), ('wrong_endpoint', 'interior_sham')]:
            a, b = cells[prefix+'/'+left], cells[prefix+'/'+right]
            require(a.keys() == b.keys(), 'Position contrast must pair identical query images')
            first = next(iter(a.values()))
            contrasts[left+'-minus-'+right] = {name: None if first[name] is None else
                statistics.mean(a[query][name]-b[query][name] for query in sorted(a)) for name in first}
        output['paired_contrasts'][prefix] = contrasts
    return output


def verify_position(recomputed, reference):
    def compare(left, right, path=''):
        if isinstance(left, dict):
            require(left.keys() == right.keys(), 'Position fields differ: '+path)
            for key in left:
                compare(left[key], right[key], path+'/'+key)
        elif isinstance(left, (int, float)):
            require(math.isclose(left, right, abs_tol=1e-12, rel_tol=0), 'Position value differs: '+path)
        else:
            require(left == right, 'Position value differs: '+path)
    for field in recomputed:
        compare(recomputed[field], reference[field], field)
    return dict(records=recomputed['raw_records'], cells=len(recomputed['cells']), all_values_match=True)
