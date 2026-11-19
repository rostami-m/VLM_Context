"""Run the matched-budget five-task continual study on the declared streams."""
from argparse import Namespace
from pathlib import Path
import gc

from vlm_context.io import read_json, require, source_path, write_json
from vlm_context.kernels.continual.contract import ARMS, POLICIES, validate_data
from vlm_context.kernels.continual.native import NativeRun


def run_streams(config, args, *, controller=False):
    root = Path(args.data_root) / 'continual'
    data = read_json(root / 'image_manifest.json')
    schedule = read_json(source_path('data/protocols/continual/stream_schedule.json.gz'))
    targets = read_json(source_path('data/protocols/continual/continual_target_contract.json'))
    validate_data(data, schedule, targets)
    modes = POLICIES if controller else ARMS
    arms = config.get('arms', list(modes))
    orders = config.get('orders', [0] if controller else [0, 1, 2])
    seeds = config.get('seeds', [0, 1, 2])
    require(set(arms) <= set(modes), 'Unknown continual arm or controller policy')
    require(set(orders) <= ({0} if controller else {0, 1, 2}), 'Task-order scope differs from the paper')
    require(set(seeds) <= {0, 1, 2} and arms and orders and seeds, 'Empty or undeclared stream grid')
    if args.smoke:
        arms, orders, seeds = arms[:1], orders[:1], seeds[:1]
    implementation = NativeRun
    if controller:
        from vlm_context.kernels.controller.native import ControllerRun
        implementation = ControllerRun
    index = []
    for arm in arms:
        for order in orders:
            for seed in seeds:
                stream = f'order_{order}_seed_{seed}'
                output = Path(args.output) / arm / stream
                run_config = {'experiment': 'Q18' if controller else 'Q17', 'arm': arm,
                    'stream_id': stream, 'mode': 'smoke' if args.smoke else 'full',
                    'snapshot': str(Path(args.model_dir).resolve()), 'device': args.device,
                    'data': str(root.resolve()), 'gate_threshold': config.get('gate_threshold', 0.25)}
                if controller:
                    require(run_config['gate_threshold'] == .25, 'Paper controller uses the fixed development-selected threshold .25')
                experiment = implementation(Namespace(output=output), run_config, data, schedule, targets)
                experiment.execute()
                index.append({'arm': arm, 'stream': stream, 'result': str(output / 'result.json')})
                del experiment
                gc.collect()
    write_json(Path(args.output) / 'index.json', {'family': 'controller' if controller else 'continual', 'runs': index,
        'smoke': args.smoke, 'inference_precision': 'float32' if controller else 'bfloat16'})
    return index


def run(config, args):
    return run_streams(config, args)
