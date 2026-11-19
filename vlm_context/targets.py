"""Exact target inventories for the diagnostic objective/location study."""
from __future__ import annotations
import re
from .io import require


PROJECTION_ORDER = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                    'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')



SPARSE = {'early': (0, 4, 8), 'middle': (9, 13, 17), 'late': (19, 23, 27)}



EXPECTED = {'connector': 135168, 'early': 122880, 'middle': 122880, 'late': 122880,
            'historical_capped_all': 2992128, 'uncapped_all': 9232384,
            'matched_all_depth': 122880}



RANK2_BLOCKS = (0, 2, 4, 7, 9, 12, 14, 17, 19, 22, 24, 27)



def arm_specs(include_matched=False):
    result = {}
    for objective, loci in [('natural', ['connector', 'late']), ('episode_remap',
        ['connector', 'early', 'middle', 'late', 'historical_capped_all', 'uncapped_all'] +
        (['matched_all_depth'] if include_matched else []))]:
        for locus in loci:
            result[locus + '_' + objective] = {'objective': objective, 'locus': locus,
                'expected_trainable_parameters': EXPECTED[locus]}
    return result



def target_spec(locus):
    if locus == 'connector':
        names = ['visual.merger.mlp.0', 'visual.merger.mlp.2']
    elif locus in SPARSE:
        q, up, v = SPARSE[locus]
        names = [f'model.layers.{q}.self_attn.q_proj', f'model.layers.{up}.mlp.up_proj',
                 f'model.layers.{v}.self_attn.v_proj']
    elif locus in ('historical_capped_all', 'uncapped_all'):
        names = [f'model.layers.{i}.{projection}' for i in range(28) for projection in PROJECTION_ORDER]
        if locus == 'historical_capped_all': names = names[:64]
    elif locus == 'matched_all_depth':
        names = [f'model.layers.{i}.self_attn.q_proj' for i in range(28)]
    else:
        raise ValueError('undeclared target locus')
    ranks = {name: 8 for name in names}
    if locus == 'matched_all_depth':
        ranks = {name: (2 if int(name.split('.')[2]) in RANK2_BLOCKS else 1) for name in names}
    return {'names': names, 'rank_by_target': ranks,
            'alpha_by_target': {name: 2 * ranks[name] for name in names}}



def expected_linear_shapes():
    """Declared architecture contract; actual module dimensions are checked at load."""
    result = {'visual.merger.mlp.0': (5120, 5120), 'visual.merger.mlp.2': (5120, 1536)}
    dimensions = [(1536,1536), (1536,256), (1536,256), (1536,1536),
                  (1536,8960), (1536,8960), (8960,1536)]
    for block in range(28):
        for projection, dims in zip(PROJECTION_ORDER, dimensions):
            result[f'model.layers.{block}.{projection}'] = dims
    return result



def target_inventory(linear_shapes, locus):
    spec = target_spec(locus)
    expected = expected_linear_shapes()
    if locus == 'historical_capped_all':
        decoder_names=[name for name in linear_shapes if re.fullmatch(r'model\.layers\.[0-9]+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)',name)]
        require(decoder_names==target_spec('uncapped_all')['names'],'actual named decoder order differs from historical cap reconstruction')
    require(all(name in linear_shapes for name in spec['names']), 'required exact Linear target absent')
    modules = []
    for name in spec['names']:
        dims = tuple(linear_shapes[name])
        require(dims == expected[name], 'actual target dimension differs from pinned architecture')
        rank = spec['rank_by_target'][name]
        modules.append({'name': name, 'in_features': dims[0], 'out_features': dims[1], 'rank': rank,
                        'alpha': 2 * rank, 'expected_parameters': rank * sum(dims)})
    count = sum(row['expected_parameters'] for row in modules)
    require(count == EXPECTED[locus], 'actual parameter budget differs from declared target recipe')
    return {'locus': locus, 'modules': modules, 'trainable_parameters': count,
            'target_regex': '^(?:' + '|'.join(re.escape(name) for name in spec['names']) + ')$',
            'selection': 'explicit ordered names, never dynamic suffix selection or silent cap',
            'rank_by_target': spec['rank_by_target'], 'alpha_by_target': spec['alpha_by_target'],
            'capacity_matched_to_sparse_language_arms': locus in (*SPARSE, 'matched_all_depth'),
            'projection_mix_identical_to_sparse': locus in SPARSE,
            'historical_scope': 'first 64 decoder Linear modules in pinned q/k/v/o/gate/up/down order; new objective/data/runtime'
                                if locus == 'historical_capped_all' else None}
