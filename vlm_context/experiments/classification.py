"""Frozen, objective/location, joint-target and equal-budget depth experiments."""
from copy import deepcopy
from pathlib import Path
from ..io import read_json, write_json, source_path, require, sha256
from ..data import image_loader, NAMES
from ..qwen import QwenEvaluator, resolve_runtime, make_evaluator, MODEL_ID, MODEL_REVISION
from ..adaptation import fit, diagnostic_ledger
from ..explicit import expand_variant


def scheduled(config, args):
    """Return the accepted episodes and target recipe without loading a model."""
    family = config['study']
    seed = args.seed
    task = args.task or config.get('task', 'shapes4')
    arm = args.arm or config.get('arm', 'frozen')
    mode = args.mode
    if family == 'q05':
        if mode == 'train':
            protocol = read_json(source_path('data/protocols/q05/fit_protocol.json'))['protocol']
            # Training arm keys omit task; evaluation arm keys include it.
            key = arm.removeprefix(task + '_')
            require(key in protocol['arms'], 'Unknown Q05 fit arm: ' + key)
            spec = protocol['arms'][key]
            schedule = read_json(source_path('data/protocols/q05/training_schedule.json.gz'))
            episodes = schedule[task][str(seed)]['training'][spec['objective']]
            ledger, recipe = diagnostic_ledger(spec['locus']), protocol['training']
        else:
            protocol = read_json(source_path('data/protocols/q05/prepared_protocol.json'))['protocol']
            key = arm if arm.startswith(task + '_') else task + '_' + arm
            require(arm == 'frozen' or key in protocol['arms'], 'Unknown Q05 arm: ' + arm)
            schedule = read_json(source_path('data/protocols/q05/episode_schedule.json.gz'))
            episodes = [e for e in schedule['smoke' if args.smoke else 'evaluation']
                        if e['task'] == task and e['seed'] == seed]
            ledger = recipe = None
    else:
        protocol = read_json(source_path('data/protocols/core/protocol.json'))
        recipes = read_json(source_path('data/protocols/core/target_recipes.json'))['recipes']
        if mode == 'train':
            require(task == 'shapes4', 'The accepted joint/depth fits are Shapes source-only')
            if arm == 'unrestricted_q05_natural':
                episodes = read_json(source_path('data/protocols/q05/training_schedule.json.gz'))['shapes4'][str(seed)]['training']['natural']
                ledger = diagnostic_ledger('uncapped_all')
                recipe = read_json(source_path('data/protocols/q05/fit_protocol.json'))['protocol']['training']
            elif family == 'depth':
                require(arm in recipes and (arm.startswith('window_') or arm == 'connector_r8_separate_budget'),
                        'Select a depth window recipe or connector_r8_separate_budget')
                episodes = read_json(source_path('data/protocols/core/depth_schedule.json.gz'))['training'][str(seed)]
                if isinstance(episodes, dict): episodes = episodes['episode_remap']
                ledger, recipe = recipes[arm], {**protocol['training'], 'steps': 150}
            else:
                require(arm in protocol['joint_arms'], 'Unknown joint arm: ' + arm)
                spec = protocol['joint_arms'][arm]
                episodes = read_json(source_path('data/protocols/core/training_schedule.json.gz'))[str(seed)][spec['objective']]
                ledger, recipe = recipes[spec['locus']], protocol['training']
        else:
            schedule = read_json(source_path('data/protocols/core/transfer_schedule.json.gz'))
            variants = schedule['variants']
            if family == 'depth':
                indices = read_json(source_path('data/protocols/core/depth_schedule.json.gz'))['transfer_variant_indices']
                variants = [variants[i] for i in indices]
            else:
                # Retained cells are the precise manuscript evaluation subset.
                cells = read_json(source_path('data/protocols/core/retained_cells.json'))
                accepted_arm = 'frozen_joint' if arm == 'frozen' else arm
                cells = [c for c in cells if c['arm'] == accepted_arm and c['seed'] == seed and c['task'] == task]
                require(cells, 'No retained paper cell for that arm/task/seed')
                allowed = {(c['k'], condition) for c in cells for condition in c['conditions']}
                variants = [v for v in variants if (schedule['contexts'][v['context_id']]['k'], v['condition']) in allowed]
            episodes = [expand_variant(schedule['contexts'][v['context_id']], v) for v in variants
                if schedule['contexts'][v['context_id']]['task'] == task and
                   schedule['contexts'][v['context_id']]['episode_seed'] == seed]
            ledger = recipe = None
    require(episodes, 'Selected schedule is empty')
    episodes = deepcopy(episodes)
    for e in episodes:
        e.setdefault('support_labels', [e['label_space'][s['class_id']] for s in e['supports']])
        e.setdefault('target_label', e['label_space'][e['query']['class_id']])
    if args.smoke:
        # Technical smoke is a prefix only and must never be called a paper rerun.
        if mode == 'train':
            episodes = episodes[:2]
        else:
            zero = next((e for e in episodes if e['k'] == 0 and e['condition'] == 'standard'), episodes[0])
            remapped = [e for e in episodes if e['condition'] in ('episode_remap', 'remap_seen_mapping', 'remap_heldout_mapping')]
            require(remapped, 'Technical smoke requires a remapped demonstration episode')
            supported = max(remapped, key=lambda e: e['k'])
            require(supported['k'] > 0, 'Technical smoke must cover multiple images')
            episodes = [zero, supported]
    return task, arm, episodes, ledger, recipe


def validate_checkpoint(config, args, task, arm):
    """Bind adapter evaluation to its completed source fit before loading weights."""
    require(arm == 'frozen' or args.checkpoint, 'Adapted evaluation requires --checkpoint')
    require(arm != 'frozen' or not args.checkpoint, 'Frozen evaluation cannot load an adapter')
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        run = read_json(checkpoint.parent / 'run.json')
        fit_receipt = read_json(checkpoint.parent / 'fit_complete.json')
        fit_runtime = read_json(checkpoint.parent / 'runtime.json')['runtime']
        require(fit_runtime['model_id'] == MODEL_ID and fit_runtime['model_revision'] == MODEL_REVISION,
                'Checkpoint was fitted with a different base model')
        expected_task = task if config['study'] == 'q05' else 'shapes4'
        expected_arm = arm.removeprefix(task + '_').removeprefix('q05_')
        require(run['mode'] == 'train' and run['task'] == expected_task and run['seed'] == args.seed
                and run['arm'].removeprefix(expected_task + '_') == expected_arm
                and (not run['smoke_only'] or args.smoke)
                and run['episodes'] == fit_receipt['completed_steps'], 'Checkpoint arm/seed/source task differs from the evaluation contract')
        expected_steps = 150 if config['study'] in ('q05', 'depth') or arm.startswith('q05_') or arm == 'unrestricted_q05_natural' else 250
        require(fit_receipt['completed_steps'] == (2 if run['smoke_only'] else expected_steps),
                'Checkpoint does not match the fixed training budget')
        require({p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()} == fit_receipt['checkpoint_files'],
                'Checkpoint changed after the completed fit')


def run(config, args):
    task, arm, episodes, ledger, recipe = scheduled(config, args)
    out = Path(args.output)
    require(not out.exists() or not any(out.iterdir()), 'Use a fresh output directory')
    loader = image_loader(args.data_root, task)
    if args.mode == 'train':
        runtime = resolve_runtime(args.model_dir, args.device)
        require(runtime['model_id'] == MODEL_ID and runtime['model_revision'] == MODEL_REVISION, 'Classification study requires the pinned Qwen2-VL-2B model')
        evaluator = QwenEvaluator(runtime, seed=args.seed)
        evaluator.identity['runtime'] = runtime
        result = fit(evaluator, episodes, loader, ledger, recipe, out)
    else:
        validate_checkpoint(config, args, task, arm)
        evaluator = make_evaluator(args.model_dir, args.device, args.seed, args.checkpoint)
        require(evaluator.identity['runtime']['model_id'] == MODEL_ID and evaluator.identity['runtime']['model_revision'] == MODEL_REVISION,
                'Classification study requires the pinned Qwen2-VL-2B model')
        write_json(out / 'runtime.json', evaluator.identity)
        counts = {}
        for index, episode in enumerate(episodes):
            record = evaluator.evaluate(episode, loader, NAMES[task], verify=args.smoke or index == 0)
            record.update(arm=arm, fit_seed=args.seed, study=config['study'], smoke_only=args.smoke)
            write_json(out / 'records' / f'{index:06d}.json', record)
            group = (episode['k'], episode['condition'])
            counts.setdefault(group, [0, 0])
            counts[group][0] += episode['label_space'][record['scores']['sum_prediction']] == episode['target_label']
            counts[group][1] += 1
            if index % 20 == 0: print(f'evaluated {index + 1}/{len(episodes)}', flush=True)
        result = {'cells': [{'k': k, 'condition': c, 'correct': a, 'n': n, 'accuracy': a / n}
                           for (k, c), (a, n) in sorted(counts.items())]}
    write_json(out / 'run.json', {'study': config['study'], 'task': task, 'arm': arm,
        'seed': args.seed, 'mode': args.mode, 'episodes': len(episodes), 'smoke_only': args.smoke,
        'precision': {'base_load': 'bfloat16', 'adapter_training': 'float32', 'evaluation': 'float32'}})
    write_json(out / 'summary.json', result)
    return result
