"""Development-only reader/vector selection followed by fixed-query confirmation."""
from pathlib import Path
from ..io import write_json, require
from ..kernels.gate.session import PolicySession
from ..kernels.inject import operations


def run(config, args):
    require(config.get('tasks') == ['shapes4', 'cifar4'] and config.get('seeds') == [0, 1, 2],
            'Inject selection requires both tasks and all three paired seeds')
    output = Path(args.output)
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh output directory')
    if args.smoke:
        return operations.smoke(PolicySession(config, args))
    evaluator = None
    eligibility = []
    for reader in operations.READERS:
        session = PolicySession(config, args, output / 'eligibility' / reader, evaluator)
        session.args.method = reader
        operations.eligibility_run(session)
        evaluator = session.evaluator
        eligibility.append(session.output / 'eligibility.json')
    reader_selection = output / 'reader_selection.json'
    operations.choose_readers(session.project, eligibility, reader_selection)
    session = PolicySession(config, args, output / 'development', evaluator)
    session.args.checkpoint = reader_selection
    operations.development(session)
    session = PolicySession(config, args, output / 'confirmation', session.evaluator)
    session.args.checkpoint = reader_selection
    session.args.selection = output / 'development' / 'selection.json'
    result = operations.confirmation(session)
    write_json(output / 'index.json', {'reader_selection': 'reader_selection.json',
        'vector_selection': 'development/selection.json', 'confirmation': 'confirmation/summary.json',
        'final_queries_used_for_selection': False, 'smoke_only': False})
    return result
