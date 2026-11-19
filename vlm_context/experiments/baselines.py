"""Same-prompt classifier comparisons, DARA and the frozen vision-feature baseline."""
from pathlib import Path
from types import SimpleNamespace
import math
from ..io import require, read_json, write_json, source_path, sha256
from ..data import image_loader, load_manifest
from ..qwen import make_evaluator, QwenEvaluator, resolve_runtime, MODEL_ID, MODEL_REVISION
from ..baseline_scoring import evaluate_baseline, neutral_episode, neutral_loader, calibrate_panel
from ..kernels.gate.common import score, validate_score, digest
from ..kernels.inject.hooks import classification_renderer
from ..kernels.baseline import dara

class BaselineSession:
    def __init__(self, config, args):
        self.config, self.args = config, SimpleNamespace(**vars(args))
        self.args.condition = config.get('objective', 'episode_remap')
        self.output = Path(args.output)
        require(not self.output.exists() or not any(self.output.iterdir()), 'Use a fresh output directory')
        self.runtime = resolve_runtime(args.model_dir, args.device)
        require(self.runtime['model_id'] == MODEL_ID and self.runtime['model_revision'] == MODEL_REVISION,
                'The shared-prompt study requires the pinned Qwen2-VL-2B model')
        self.runtime_sha = digest(self.runtime)
        self.schedule = read_json(source_path('data/protocols/interventions/baseline_schedule.json.gz'))
        self.bundle_sha = digest(self.schedule)
        self.evaluator = None
        self.records = []

    def start_model(self):
        if self.evaluator is None:
            if self.args.mode == 'train':
                self.evaluator = QwenEvaluator(self.runtime, seed=self.args.seed)
            else:
                checkpoint = self.args.checkpoint if self.config.get('method') not in ('dara', 'dara_natural', 'dara_episode_remap') else None
                self.evaluator = make_evaluator(self.args.model_dir, self.args.device, self.args.seed, checkpoint)
                self.runtime = self.evaluator.identity['runtime']
            write_json(self.output / 'runtime.json', self.evaluator.identity)
        return self.evaluator

    def loader(self, task):
        return load_manifest(self.args.data_root, task), image_loader(self.args.data_root, task)

    def save(self, key, value):
        require(key not in self.records, 'Duplicate record')
        write_json(self.output / 'records' / (key + '.json'), value)
        self.records.append(key)

    def finish(self, details):
        value = {'records': len(self.records), **details}
        write_json(self.output / 'completion.json', value)
        return value


def run(config, args):
    session = BaselineSession(config, args)
    method = config.get('method', 'frozen')
    if args.arm: method = args.arm
    # Keep checkpoint loading tied to the selected method, including CLI override.
    session.config = {**config, 'method': method}
    if args.mode == 'train':
        require(method.startswith('dara'), 'Use the q05/core configurations to fit LoRA readers')
        session.args.condition = method.removeprefix('dara_') if method != 'dara' else config.get('objective', 'episode_remap')
        session.start_model()
        return session.finish({'fit': dara.train_steps(session, smoke=args.smoke)})
    if method in ('frozen','cama','vision_1nn'):
        require(not args.checkpoint, 'Frozen-model comparisons cannot load an adapter checkpoint')
    elif not method.startswith('dara'):
        require(args.checkpoint, 'Adapted baseline requires a completed source-only checkpoint')
        checkpoint = Path(args.checkpoint)
        source_run = read_json(checkpoint.parent / 'run.json')
        fit_receipt = read_json(checkpoint.parent / 'fit_complete.json')
        fit_runtime = read_json(checkpoint.parent / 'runtime.json')['runtime']
        require(fit_runtime['model_id'] == MODEL_ID and fit_runtime['model_revision'] == MODEL_REVISION,
                'Baseline checkpoint was fitted with a different base model')
        expected_arm = {'joint':'joint_uncapped_r16_episode_remap',
                        'late':'late_parameter_matched_r34_35_episode_remap'}.get(method, method)
        require(source_run['mode'] == 'train' and source_run['task'] == 'shapes4'
                and source_run['arm'] == expected_arm and source_run['seed'] == args.seed
                and (not source_run['smoke_only'] or args.smoke)
                and source_run['episodes'] == fit_receipt['completed_steps'], 'Baseline checkpoint has a different objective/seed/source task')
        require({p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()} == fit_receipt['checkpoint_files'],
                'Baseline checkpoint bytes changed')
    session.start_model()
    if method == 'vision_1nn':
        from ..kernels.baseline.features import features
        return features(session, smoke_only=args.smoke)
    require(method in ('frozen','cama','joint','late','connector_natural','connector_episode_remap',
                       'uncapped_all_episode_remap','unrestricted_q05_natural','dara_natural','dara_episode_remap'),
            'Unknown shared-prompt baseline')
    if method not in ('frozen','cama'):
        require(args.checkpoint, 'Adapted baseline evaluation requires --checkpoint')
    factors = None
    if method.startswith('dara_'):
        checkpoint = read_json(args.checkpoint)
        require(checkpoint['objective'] == method.removeprefix('dara_') and checkpoint['seed'] == args.seed,
                'DARA objective/seed mismatch')
        require(checkpoint['completed_steps'] == 150 and checkpoint['recipe'] == dara.RECIPE and
                not checkpoint['technical_only'], 'Incomplete DARA checkpoint')
        factors = dara.factor_tensor(session, checkpoint['factors'])
    task = args.task or config.get('task', 'cifar4')
    panels = [p for p in session.schedule['panels'] if p['episodes'][0]['task'] == task and p['episodes'][0]['seed'] == args.seed]
    require(len(panels) == 6, 'Expected six paired baseline panels')
    _, loader = session.loader(task)
    summaries = []
    for panel in panels[:1] if args.smoke else panels:
        rows = []
        for index, ep in enumerate(panel['episodes'][:2] if args.smoke else panel['episodes']):
            if method in ('frozen','cama'):
                result = evaluate_baseline(session.evaluator, session.runtime, ep, loader, method, verify=index == 0)
            elif factors is not None:
                result = dara.scored(session, ep, loader, factors, native=index == 0)
            else:
                result = score(session, ep, loader, native=index == 0, renderer=classification_renderer)
            validate_score(result, ep)
            session.save(f'{panel["panel_id"]}_{index:03d}', {'result': result, 'method': method})
            rows.append(result)
        calibration = None
        if method == 'frozen' and panel['episodes'][0]['condition'] == 'standard' and not args.smoke:
            ep = neutral_episode(panel['episodes'][0])
            neutral = evaluate_baseline(session.evaluator, session.runtime, ep, neutral_loader(loader), 'frozen')
            session.save(panel['panel_id'] + '_neutral', {'result': neutral})
            calibration = calibrate_panel(panel, rows, neutral)
        gold = [r['episode']['query']['class_id'] for r in rows]
        predictions = [r['scores']['sum_prediction'] for r in rows]
        summaries.append({'panel_id': panel['panel_id'], 'accuracy': sum(a == b for a,b in zip(predictions,gold))/len(rows),
            'n': len(rows), 'predictions': predictions, 'gold': gold, 'calibration': calibration,
            'query_ids': [r['episode']['query']['image_id'] for r in rows], 'seconds': math.fsum(r['seconds'] for r in rows)})
    write_json(session.output / 'summary.json', {'method': method, 'panels': summaries, 'smoke_only': args.smoke,
        'prompt': 'CAMA A-only classification', 'calibration': 'standard-label CC and unlabeled transductive BC',
        'evaluation_dtype': 'float32', 'base_load_dtype': 'bfloat16'})
    return session.finish({'panels': len(summaries)})
