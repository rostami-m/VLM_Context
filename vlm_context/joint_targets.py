"""Exact parameter ledgers, mixed-rank matching and omission definitions."""
from __future__ import annotations
from collections import Counter
import re
from .io import require


def target_ledger(name, modules, ranks):
    require(len(modules) == len({r['name'] for r in modules}), 'duplicate target')
    rows = []
    for module in modules:
        rank = ranks if type(ranks) is int else ranks[module['name']]
        require(type(rank) is int and 0 < rank <= min(module['in_features'], module['out_features']), 'invalid rank')
        row = {k: module[k] for k in ['name', 'in_features', 'out_features']}
        row.update(rank=rank, alpha=rank * 2,
                   trainable_parameters=rank * (row['in_features'] + row['out_features']))
        rows.append(row)
    return {'name': name, 'modules': rows, 'module_count': len(rows),
        'trainable_parameters': sum(r['trainable_parameters'] for r in rows),
        'rank_histogram': dict(Counter(str(r['rank']) for r in rows)),
        'target_regex': '^(?:' + '|'.join(re.escape(r['name']) for r in rows) + ')$',
        'rank_pattern': {r['name']: r['rank'] for r in rows},
        'alpha_pattern': {r['name']: r['alpha'] for r in rows},
        'measurement': 'prior audited tensor-header dimensions; not runtime injection'}



def validate_instantiated_targets(expected, actual):
    """Pure-data guard for a future runner's actual trainable tensor inventory.

    A caller must obtain actual from the instantiated model; passing a predicted
    inventory here is a unit test only and cannot satisfy a runtime gate.
    """
    rows = {r['name']:r for r in actual}
    require(len(rows) == len(actual), 'duplicate actual trainable tensor')
    wanted = {}
    for module in expected['modules']:
        wanted[module['name']+'.lora_A.default.weight'] = [module['rank'],module['in_features']]
        wanted[module['name']+'.lora_B.default.weight'] = [module['out_features'],module['rank']]
    require(set(rows) == set(wanted), 'extra or missing trainable tensors')
    for name,shape in wanted.items():
        require(rows[name]['shape'] == shape and rows[name]['dtype'] == 'torch.float32'
                and rows[name]['numel'] == shape[0]*shape[1], 'actual rank/shape/dtype mismatch')
    require(sum(r['numel'] for r in rows.values()) == expected['trainable_parameters'], 'actual parameter count mismatch')
    return True



def make_recipes(audit):
    previous = {r['name']: r for r in audit['legacy_and_joint_reference']}
    joint = previous['joint_uncapped_r16']['modules']
    late = previous['legacy_late_r16']['modules']
    rank_map = {r['name']: 34 for r in late}
    # Exact residual from uniform rank34: 734*256 parameters.
    # 16 MLP increments (41 units), 3 q/o increments (12), 6 k/v (7).
    # Minimize count of increments and spread them deterministically by depth.
    groups = [([r for r in late if '.mlp.' in r['name']], 16),
              ([r for r in late if re.search(r'\.(q|o)_proj$', r['name'])], 3),
              ([r for r in late if re.search(r'\.(k|v)_proj$', r['name'])], 6)]
    for candidates, count in groups:
        for index in [i * len(candidates) // count for i in range(count)]:
            rank_map[candidates[index]['name']] += 1
    recipes = {'joint_uncapped_r16': target_ledger('joint_uncapped_r16', joint, 16),
        'late_parameter_matched_r34_35': target_ledger('late_parameter_matched_r34_35', late, rank_map)}
    for omitted in ['connector', 'early', 'mid']:
        def keep(row):
            if row['name'].startswith('visual.'):
                return omitted != 'connector'
            block = int(row['name'].split('.')[2])
            return not (omitted == 'early' and block < 10 or omitted == 'mid' and 10 <= block < 19)
        label = 'joint_without_' + omitted + '_r16'
        recipes[label] = target_ledger(label, [r for r in joint if keep(r)], 16)
    for row in audit['seven_window_inventory']:
        recipes[row['name']] = target_ledger(row['name'], row['modules'], 8)
    connector = [r for r in joint if r['name'].startswith('visual.')]
    recipes['connector_r8_separate_budget'] = target_ledger('connector_r8_separate_budget', connector, 8)
    full = recipes['joint_uncapped_r16']['trainable_parameters']
    require(full == recipes['late_parameter_matched_r34_35']['trainable_parameters'] == 12800000,
            'joint versus late parameter mismatch')
    for label in recipes:
        if label.startswith('joint_without_'):
            recipes[label]['parameter_delta_from_full_joint'] = recipes[label]['trainable_parameters'] - full
            recipes[label]['capacity_controlled_ablation'] = False
    return {'recipes': recipes,
        'legacy_reconstructions_preserved_without_retraining': audit['legacy_and_joint_reference'],
        'late_matching_disclosure': 'Same parameter count only: late blocks19-27, all seven projection types, rank34/35 versus joint connector+blocks0-18 rank16. Projection composition, per-module rank and downstream influence differ. Alpha/rank remains2.',
        'legacy_disclosure': 'Historical capped128 joint has12140544 parameters and omits block18; corrected uncapped135 has12800000. Rank16/250steps alone does not reproduce historical supervision, data or checkpoint provenance.'}
