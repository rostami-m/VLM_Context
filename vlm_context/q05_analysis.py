"""Recalculate Q05 predictions from original answers and candidate token scores."""
import gzip
import hashlib
import json

from .io import read_json, require, sha256, source_path
from .protocol import parse_diagnostics, sequence_likelihoods


def reduce_record(row):
    """Derive outcomes; never consume previously calculated predictions or flags."""
    labels = row['label_space']
    require(type(row['gold']) is int and 0 <= row['gold'] < len(labels), 'Q05 gold index outside label space')
    candidates = row['candidate_tokens']
    probabilities = row['per_token_label_log_probabilities']
    require(len(candidates) == len(probabilities) == len(labels), 'Q05 candidate count differs')
    for label, candidate, logps in zip(labels, candidates, probabilities):
        require(candidate['label'] == label, 'Q05 candidate label order differs')
        require(candidate['scored_token_ids'] == candidate['label_token_ids'] + [row['eos_token_id']],
                'Q05 answer must include exactly one appended end token')
        require(row['eos_token_id'] not in candidate['label_token_ids'], 'Q05 end token inside label')
        require(len(logps) == len(candidate['scored_token_ids']), 'Q05 token-score alignment differs')
    scores = sequence_likelihoods(probabilities)
    parsed = parse_diagnostics(row['greedy_answer'], labels, natural_labels=row['natural_labels'])
    predictions = dict(sum=scores['sum_prediction'], mean=scores['mean_prediction'],
                       strict=parsed['prediction'], historical=parsed['legacy_prediction'])
    result = {key: row[key] for key in ('task', 'arm', 'condition', 'k', 'episode_seed', 'fit_seed',
                                      'image_id', 'gold', 'label_space')}
    result.update(predictions=predictions,
        invalid={key: value is None for key, value in predictions.items()},
        correct={key: int(value == row['gold']) for key, value in predictions.items()},
        ambiguous=parsed['status'] == 'ambiguous_labels',
        natural_label_emission=parsed['natural_label_emission'],
        option_echo=parsed['status'] == 'option_echo',
        parser_disagreement=predictions['strict'] != predictions['historical'],
        scorer_disagreement=predictions['sum'] != predictions['mean'])
    return result


def reconstruct_q05(path=None):
    """Read the compact original records and return independently derived rows."""
    manifest = read_json(source_path('data/analysis/q05_manifest.json')) if path is None else None
    path = source_path(manifest['output']) if manifest is not None else path
    if manifest is not None:
        require(sha256(path) == manifest['output_sha256'], 'Q05 compressed evidence hash differs')
    payload_hash = hashlib.sha256()
    rows = []
    with gzip.open(path, 'rb') as handle:
        for line in handle:
            payload_hash.update(line)
            rows.append(reduce_record(json.loads(line)))
    require(bool(rows), 'No Q05 raw records')
    if manifest is not None:
        require(len(rows) == manifest['records'], 'Q05 raw record inventory differs')
        require(payload_hash.hexdigest() == manifest['decompressed_sha256'], 'Q05 decompressed evidence hash differs')
    return rows


def verify_q05_outcomes(rows, reference):
    """Check image-level pairing and outcomes, not only aggregate agreement."""
    identity_fields = ('task', 'arm', 'condition', 'k', 'episode_seed', 'fit_seed', 'image_id')
    def indexed(values):
        result = {tuple(row[key] for key in identity_fields): row for row in values}
        require(len(result) == len(values), 'Duplicate Q05 raw/reference identity')
        return result
    actual, expected = indexed(rows), indexed(reference)
    require(actual.keys() == expected.keys(), 'Q05 raw/reference identity coverage differs')
    for identity, row in actual.items():
        require(all(value == expected[identity][key] for key, value in row.items()),
                'Q05 raw/reference outcome differs: ' + str(identity))
    return {'per_image_outcomes_checked': len(actual), 'all_per_image_outcomes_match': True}
