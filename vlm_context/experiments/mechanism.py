"""Paired residual patches, attention cuts and descriptive attention mass."""
from pathlib import Path
import math
from ..io import require, read_json, write_json
from ..kernels.gate.session import PolicySession
from ..kernels.mechanism import operations as op


def run(config, args):
    config = {**config, 'tasks': ['shapes4']}
    phase = config.get('phase', 'all')
    session = PolicySession(config, args, Path(args.output) / ('smoke' if args.smoke else 'development'))
    if args.smoke:
        return op.smoke(session)
    if phase == 'attention':
        return op.attention_descriptive(session)
    if phase in ('all', 'development'):
        op.development(session)
    if phase == 'development': return
    selection_path = config.get('selection', str(Path(args.output) / 'development' / 'selection.json'))
    selected = read_json(selection_path)
    require(selected['final_queries_used'] is False and selected['layer'] in op.LAYERS, 'Invalid development selection')
    means = {int(k): v for k,v in selected['mean_correct_label_logp_restoration'].items()}
    require(selected['layer'] == min(op.LAYERS, key=lambda layer: (-means[layer], layer)), 'Layer selection rule changed')
    session = PolicySession(config, args, Path(args.output) / 'confirmation', evaluator=session.evaluator)
    session.start_model()
    manifest, loader = session.loader('shapes4')
    rows = []
    for block in op.blocks(session):
        for index, query in enumerate(block['final_query_ids']):
            outcomes = op.trace(session, block, manifest, loader, query, 'confirmation_query', selected['layer'],
                                f'final_s{block["seed"]}_q{index:03d}')
            rows.append({'block_id': block['block_id'], 'query_id': query, 'outcomes': outcomes})
    summaries = {method: {'accuracy': math.fsum(r['outcomes'][method]['correct'] for r in rows)/len(rows),
        'mean_correct_label_logp': math.fsum(r['outcomes'][method]['correct_label_logp'] for r in rows)/len(rows)}
        for method in op.METHODS}
    write_json(session.output / 'summary.json', {'layer': selected['layer'], 'summaries': summaries,
        'paired_rows': rows, 'inference_unit': '80 fixed images repeated across three context seeds',
        'interpretation': 'contributing path; descriptive intervention'})
    return session.finish({'paired_queries': len(rows)})
