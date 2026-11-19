"""Develop a threshold on reserved images, then evaluate prospective Gate."""
from pathlib import Path

from vlm_context.io import require, write_json
from vlm_context.kernels.gate.session import PolicySession
from vlm_context.kernels.gate import operations


def run(config, args):
    phase = config.get('phase', 'all')
    require(phase in ('all', 'development', 'confirmation'), 'Unknown Gate phase')
    if args.smoke:
        session = PolicySession(config, args, Path(args.output) / 'smoke')
        return operations.smoke(session)
    results, evaluator = {}, None
    if phase in ('all', 'development'):
        session = PolicySession(config, args, Path(args.output) / 'development')
        results['development'] = operations.development(session)
        evaluator = session.evaluator
    if phase in ('all', 'confirmation'):
        session = PolicySession(config, args, Path(args.output) / 'confirmation', evaluator=evaluator)
        session.args.selection = config.get('selection', str(Path(args.output) / 'development' / 'selection.json'))
        results['confirmation'] = operations.confirmation(session)
    write_json(Path(args.output) / 'index.json', {'family': 'gate', 'phases': results})
    return results
