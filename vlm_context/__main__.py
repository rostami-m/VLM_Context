"""Portable command line; heavy model dependencies load only for model runs."""
import argparse
import importlib
import json
from pathlib import Path
from .io import read_json, write_json, source_path, require


def main(argv=None):
    parser = argparse.ArgumentParser(prog='vlm-context')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Run one configured experiment family')
    run.add_argument('--config', required=True)
    run.add_argument('--model-dir', required=True)
    run.add_argument('--data-root', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    run.add_argument('--mode', choices=['train', 'evaluate'], default='evaluate')
    run.add_argument('--task')
    run.add_argument('--arm')
    run.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
    run.add_argument('--checkpoint')
    run.add_argument('--smoke', action='store_true', help='Tiny technical run, never a paper result')
    data = sub.add_parser('prepare-data', help='Reconstruct pinned image banks and validate pixel hashes')
    data.add_argument('--task', choices=['shapes4', 'cifar4', 'fashion4', 'continual', 'vqav2', 'textvqa', 'scienceqa', 'rule'], required=True)
    data.add_argument('--data-root', required=True)
    data.add_argument('--cache-dir')
    data.add_argument('--parquet-root')
    fetch = sub.add_parser('fetch-model', help='Download the pinned Qwen2-VL-2B snapshot')
    fetch.add_argument('--cache-dir')
    fetch.add_argument('--size', choices=['2B', '7B'], default='2B')
    regen = sub.add_parser('regenerate', help='Recompute numerical summaries and figures from the released evidence')
    regen.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'run':
        config = read_json(args.config)
        name = config['family']
        require(name.isidentifier(), 'Invalid experiment family')
        module = importlib.import_module('.experiments.' + name, 'vlm_context')
        module.run(config, args)
    elif args.command == 'prepare-data':
        from .data import prepare_shapes, prepare_from_manifest
        if args.task == 'shapes4':
            prepare_shapes(args.data_root)
        elif args.task == 'continual':
            from .kernels.continual.materialize import materialize
            materialize(args.data_root, parquet_root=args.parquet_root)
        elif args.task in ('vqav2', 'textvqa', 'scienceqa', 'rule'):
            from .open_tasks import prepare_data
            prepare_data(args.data_root, args.task, cache_dir=args.cache_dir, parquet_root=args.parquet_root)
        else:
            prepare_from_manifest(args.task,
                source_path('data/protocols/datasets/' + args.task + '_manifest.json.gz'),
                args.data_root, cache_dir=args.cache_dir)
    elif args.command == 'fetch-model':
        from huggingface_hub import snapshot_download
        spec = read_json(source_path('configs/qwen2_snapshot.json' if args.size == '2B' else 'configs/qwen2_7b_snapshot.json'))
        path = snapshot_download(spec['model_id'], revision=spec['revision'], cache_dir=args.cache_dir)
        print(path)
    else:
        from .regenerate import regenerate
        regenerate(args.output)

if __name__ == '__main__':
    main()
