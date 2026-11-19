"""Frozen data-access and policy contract for native Q17/Q18 execution.

Standard-library only. A task identifier is bookkeeping and is never rendered.
Controller labels arrive through the nested support prefix; discarded replay
images cannot be recovered from the full manifest.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

VERSION = 'continual-native-20261106-v1'
KS = (0, 1, 2, 4, 8, 16)
ARMS = ('joint', 'late', 'all_depth', 'late_replay')
POLICIES = ('prompt_only_gate', 'fixed_schedule_consolidation', 'gate_triggered_consolidation')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    from vlm_context.io import read_json
    return read_json(path)

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

def episode(manifest, query_id, support_ids=(), seed=0):
    images = manifest['images']
    q = images[query_id]
    require(query_id not in support_ids and len(support_ids) == len(set(support_ids)), 'query leakage or duplicated support')
    supports = [images[i] for i in support_ids]
    require(all(r['role'] == 'continual_training' and r['task_id'] == q['task_id'] for r in supports), 'support outside current training task')
    require(q['content_sha256'] not in {r['content_sha256'] for r in supports}, 'pixel leakage')
    record = lambda r: {k: r[k] for k in ('image_id', 'content_sha256', 'class_id')}
    return {'protocol_version': VERSION, 'task': 'cifar100_continual', 'seed': seed,
            'condition': 'fixed_ABCD_no_task_id', 'sampler': 'frozen_nested_prefix',
            'k': len(supports), 'query': record(q), 'supports': [record(r) for r in supports],
            'label_space': list('ABCD'), 'option_order': list(range(4)), 'displayed_options': list('ABCD'),
            'supervised_label': 'ABCD'[q['class_id']]}

def validate_data(manifest, schedule, targets):
    require(len(manifest['images']) == 2360 and len(schedule['streams']) == 9, 'changed frozen data scope')
    require(schedule['task_identity_in_prompt'] is False, 'task identity cannot be in prompt')
    ids, pixels, raw = set(), set(), set()
    for role, members in manifest['roles'].items():
        for i in members:
            r = manifest['images'][i]
            require(r['role'] == role and i not in ids and r['content_sha256'] not in pixels and r['raw_rgb_sha256'] not in raw, 'overlapping data roles')
            ids.add(i); pixels.add(r['content_sha256']); raw.add(r['raw_rgb_sha256'])
    require(len(ids) == 2360, 'unassigned images')
    for stream in schedule['streams']:
        require(len(stream['task_order']) == len(set(stream['task_order'])) == 5, 'task-order drift')
        previous = []
        for index, stage in enumerate(stream['stages']):
            current = stage['current_task']
            require(current == stream['task_order'][index] and stage['replay_buffer_before'] == previous, 'replay/order discontinuity')
            released = set(stage['released_current_training_ids'])
            require(len(released) == 192 and all(manifest['images'][i]['task_id'] == current and manifest['images'][i]['role'] == 'continual_training' for i in released), 'wrong current release')
            require(set(stage['replay_buffer_after']) <= released | set(previous) and len(stage['replay_buffer_after']) <= 16, 'discarded/future replay recovery')
            for replay in (False, True):
                rows = stage['replay_reference_training_rows' if replay else 'training_rows']
                require(len(rows) == 80, 'changed step budget')
                allowed = released | (set(previous) if replay else set())
                require(all(r['query_id'] in allowed and r['output_label'] == 'ABCD'[manifest['images'][r['query_id']]['class_id']] for r in rows), 'unavailable or mislabelled training row')
                replay_ids = [r['query_id'] for r in rows if r['query_id'] not in released]
                require(len(replay_ids) == (16 if replay and index else 0), 'replay step budget changed')
            nested = stage['controller_support']['nested_support_ids']
            require(all(nested[str(k)] == nested['16'][:k] for k in KS), 'non-nested controller prefixes')
            require(set(nested['16']) <= released, 'controller uses extra labels')
            previous = stage['replay_buffer_after']
    for recipe in targets['recipes'].values():
        require(recipe['trainable_parameters'] == 12800000, 'adapter budget drift')
        require(sum((r['in_features'] + r['out_features']) * recipe['rank_pattern'][r['name']] for r in recipe['modules']) == 12800000, 'adapter budget arithmetic')
    return {'images': len(ids), 'streams': 9, 'stages': 45, 'paired_training_schedules': True, 'persistent_replay_limit': 16, 'trainable_parameters_per_arm': 12800000}

def controller_training_rows(prefix, replay, images, current_task, seed, steps=80):
    """Deterministic 20% replay allocation, using only physically retained IDs."""
    require(prefix and len(prefix) == len(set(prefix)) and len(replay) == len(set(replay)) <= 16, 'invalid controller memory')
    require(not set(prefix) & set(replay), 'current/replay overlap')
    require(all(images[i]['role'] == 'continual_training' and images[i]['task_id'] == current_task for i in prefix), 'future/evaluation/current label access')
    require(all(images[i]['role'] == 'continual_training' and images[i]['task_id'] != current_task for i in replay), 'invalid retained replay')
    require(1 <= steps <= 80, 'controller exceeds update bound')
    order = sorted(prefix, key=lambda i: digest([VERSION, 'controller-training', seed, current_task, i]))
    rows = []
    for step in range(1, steps + 1):
        is_replay = bool(replay) and step % 5 == 0
        i = replay[(step // 5 - 1) % len(replay)] if is_replay else order[(step - 1) % len(order)]
        r = images[i]
        rows.append({'step': step, 'query_id': i, 'task_id': r['task_id'], 'class_id': r['class_id'], 'output_label': 'ABCD'[r['class_id']], 'example_origin': 'replay' if is_replay else 'current'})
    return rows

def retain_replay(previous, current, images, seed):
    require(len(previous) <= 16 and not set(previous) & set(current), 'invalid replay transition')
    candidates = list(previous) + list(current)
    require(all(images[i]['role'] == 'continual_training' for i in candidates), 'evaluation data in replay')
    return sorted(candidates, key=lambda i: digest([VERSION, 'controller-replay', seed, i]))[:16]

def policy_spec():
    return {'version': VERSION, 'source': 'fresh seeded zero-effective-update joint adapter on pinned base model',
            'label_space': list('ABCD'), 'task_identity_in_prompt': False,
            'gate_threshold_source': 'Gate per-model episode_remap development threshold, fixed before final queries',
            'trigger': 'two consecutive prefix arrivals whose Gate selects less context than available',
            'fixed_schedule_prefix': 8, 'post_update_behavior': 'rerun Gate on current available prefix using updated model; cache cleared',
            'maximum_events_per_task': 1, 'updates_per_event': 80, 'retained_replay_images': 16,
            'replay_fraction_when_nonempty': .2, 'allowed_prefixes': list(KS),
            'development_protocol': 'all three arms, order_0_seed_0, development queries only; frozen rule verified before final queries',
            'final_streams': 'order 0 and seeds 0, 1, 2; separately initialized per policy; same nested prefixes and retained-label allowance',
            'final_outcomes_never_drive_trigger_or_tuning': True}

class Trigger:
    def __init__(self, policy):
        require(policy in POLICIES, 'unknown controller policy')
        self.policy, self.consecutive, self.updated = policy, 0, False

    def observe(self, available_k, selected_k):
        require(available_k in KS and selected_k in KS and selected_k <= available_k, 'invalid Gate decision')
        self.consecutive = self.consecutive + 1 if selected_k < available_k else 0
        fire = not self.updated and ((self.policy == 'fixed_schedule_consolidation' and available_k == 8) or (self.policy == 'gate_triggered_consolidation' and self.consecutive >= 2))
        if fire:
            self.updated = True
        return fire
