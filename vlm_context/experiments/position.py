"""Order-only intervention with fixed supports, labels, frequencies and images."""
from pathlib import Path
from ..io import require, read_json, write_json, source_path
from ..data import load_manifest, image_loader
from ..qwen import QwenEvaluator, resolve_runtime
from ..kernels.position import expand, CONDITIONS, validate_result
from ..kernels.position_schedule import position_triplet


def run(config, args):
    out = Path(args.output)
    require(not out.exists() or not any(out.iterdir()), 'Use a fresh output directory')
    manifest = load_manifest(args.data_root, 'shapes4')
    triplets = read_json(source_path('data/protocols/interventions/position_triplets.json.gz'))
    role = 'confirmation_query'
    if args.smoke:
        triplets = []
        for c in range(4):
            query = sorted(i for i in manifest['roles']['development'] if manifest['images'][i]['class_id'] == c)[0]
            triplets.append(position_triplet(manifest['images'], manifest['roles']['evaluation_support'], query, 0, CONDITIONS[c % 2]))
        role = 'development'
    else:
        triplets = [t for t in triplets if t['seed'] in config.get('seeds', [0, 1, 2])]
    records = expand(triplets, manifest, role)
    # This accepted experiment predates the later FP32 mechanism cohort.
    evaluator = QwenEvaluator(resolve_runtime(args.model_dir, args.device, load_dtype='bfloat16'), seed=0)
    loader = image_loader(args.data_root, 'shapes4')
    write_json(out / 'runtime.json', evaluator.identity)
    for index, record in enumerate(records):
        result = evaluator.evaluate(record['episode'], loader, manifest['class_names'], verify=args.smoke)
        validate_result(result, record, smoke=args.smoke)
        write_json(out / 'records' / (record['record_id'] + '.json'), {'record': record, 'result': result})
        if index % 20 == 0: print(f'position {index + 1}/{len(records)}', flush=True)
    value = {'records': len(records), 'triplets': len(triplets), 'smoke_only': args.smoke, 'dtype': 'bfloat16'}
    write_json(out / 'completion.json', value)
    return value
