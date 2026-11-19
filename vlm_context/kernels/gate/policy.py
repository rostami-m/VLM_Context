"""Gate threshold selection and mapping-specific vector arithmetic.

These pure functions preserve the exact grids, denominators and tie rules
used for the paper's final policy experiments.
"""
from __future__ import annotations
import math
KS=(0,1,2,4,8,16)
THRESHOLDS=(.25,.35,.45,.55,.65)

def require(value, message):
    if not value:
        raise ValueError(message)


def first_crossing(counts, threshold, probe_count=40):
    require(threshold in THRESHOLDS, 'threshold is outside the frozen grid')
    selected, observations = 0, []
    for k in KS[1:]:
        row = counts[str(k)]
        require(row['total'] == probe_count and type(row['copied']) is int and
                0 <= row['copied'] <= probe_count, 'invalid probe counts')
        rejected = row['copied'] / probe_count > threshold
        observations.append({'k': k, **row, 'rejected': rejected})
        if rejected:
            break
        selected = k
    return {'selected_k': selected, 'observations': observations,
            'probe_calls': sum(r['total'] for r in observations),
            'first_crossing_k': next((r['k'] for r in observations if r['rejected']), None)}


def select_threshold(blocks):
    """Each block carries development accuracies for the same disjoint queries."""
    require(bool(blocks), 'empty development cohort')
    candidates = []
    for threshold in THRESHOLDS:
        policies = [first_crossing(b['copy_counts'], threshold) for b in blocks]
        accuracies=[b['development_accuracy'][str(p['selected_k'])] for b,p in zip(blocks,policies)]
        require(all(0<=a<=1 and abs(a*40-round(a*40))<1e-10 for a in accuracies),'accuracy must represent40 development queries')
        score=sum(round(a*40) for a in accuracies)/(40*len(blocks))
        mean_k = math.fsum(p['selected_k'] for p in policies) / len(policies)
        candidates.append({'threshold': threshold, 'accuracy': score,
                           'mean_selected_k': mean_k, 'block_policies': policies})
    selected = min(candidates, key=lambda r: (-r['accuracy'], r['mean_selected_k'], r['threshold']))
    return {'threshold': selected['threshold'], 'candidate_summaries': candidates,
            'selected_by': 'max paired development accuracy; ties lower mean selected K, then smaller threshold'}


def mean_difference(pairs):
    require(bool(pairs) and bool(pairs[0][0]), 'empty extraction')
    width = len(pairs[0][0])
    require(all(len(a) == len(b) == width and all(math.isfinite(v) for v in a + b)
                for a, b in pairs), 'nonfinite or inconsistent vectors')
    return [math.fsum(a[i] - b[i] for a, b in pairs) / len(pairs) for i in range(width)]


def mean_state(states):
    require(bool(states) and bool(states[0]), 'empty states')
    width = len(states[0])
    require(all(len(r) == width and all(math.isfinite(v) for v in r) for r in states), 'invalid states')
    return [math.fsum(r[i] for r in states) / len(states) for i in range(width)]


def eligibility(zero_correct, supported_correct):
    """Prespecified conservative operational prerequisite, not a significance test."""
    require(len(zero_correct) == len(supported_correct) == 40, 'requires40 paired development queries')
    require(all(type(v) is bool for v in zero_correct + supported_correct), 'Boolean outcomes required')
    zero, supported = sum(zero_correct) / 40, sum(supported_correct) / 40
    return {'eligible': sum(supported_correct)>=24 and sum(supported_correct)-sum(zero_correct)>=4,
            'zero_accuracy': zero, 'supported_accuracy': supported,
            'zero_correct':sum(zero_correct),'supported_correct':sum(supported_correct),'queries':40,
            'minimum_supported_accuracy': .60, 'minimum_gain': .10,
            'interpretation': 'development operational prerequisite; no efficacy acceptance'}


def cosine_nearest(query, supports):
    """Inputs are unit vectors; stable image-ID ties independent of support order."""
    require(bool(supports), '1NN requires nonempty exact episode support')
    values = [(math.fsum(a*b for a,b in zip(query, r['unit'])), r['image_id'], r['class_id']) for r in supports]
    require(all(len(r['unit']) == len(query) for r in supports), 'feature dimensions differ')
    best = min(values, key=lambda x: (-x[0], x[1]))
    return {'class_id': best[2], 'image_id': best[1], 'cosine': best[0],
            'ties': sorted(r[1] for r in values if r[0] == best[0])}
