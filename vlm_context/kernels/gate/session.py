"""Portable execution state for the frozen-model policy experiments."""
from pathlib import Path
from types import SimpleNamespace

from vlm_context.data import load_manifest, image_loader
from vlm_context.io import read_json, require, source_path, package_root, write_json
from vlm_context.qwen import make_evaluator, resolve_runtime
from .common import digest


class PolicySession:
    def __init__(self, config, args, output=None, evaluator=None):
        self.config, self.args = config, SimpleNamespace(**vars(args))
        self.project = package_root()
        self.output = Path(output if output is not None else args.output)
        require(not (self.output / 'summary.json').exists() and not list((self.output / 'records').glob('*.json')),
                'Choose a fresh output directory; completed/partial evidence is preserved')
        blocks = read_json(source_path('data/protocols/interventions/nested_blocks.json'))
        self.blocks = [b for b in blocks if b['task'] in config.get('tasks', ['shapes4', 'cifar4'])
                       and b['seed'] in config.get('seeds', [0, 1, 2])]
        require(self.blocks, 'No policy blocks selected')
        self.manifests = {t: load_manifest(args.data_root, t) for t in {b['task'] for b in self.blocks}}
        self.runtime = resolve_runtime(args.model_dir, args.device, load_dtype='bfloat16')
        self.runtime['model_id'] = config.get('model_id', 'Qwen/Qwen2-VL-2B-Instruct')
        self.runtime['inference_dtype'] = 'float32'
        self.runtime['load_dtype'] = 'bfloat16'
        self.runtime['dtype'] = 'float32'
        self.runtime_sha = digest(self.runtime)
        self.bundle_sha = digest({'blocks': self.blocks, 'manifests': self.manifests,
                                  'model_id': self.runtime['model_id']})
        self.evaluator = evaluator
        self.records = []
        write_json(self.output / 'run_manifest.json', {'config': config, 'runtime': self.runtime,
            'episode_manifest_sha256': self.bundle_sha, 'runtime_sha256': self.runtime_sha})

    def start_model(self):
        if self.evaluator is None:
            self.evaluator = make_evaluator(self.args.model_dir, self.args.device,
                seed=0, inference_dtype='float32')
        expected = 1536 if '2B' in self.runtime['model_id'] else 3584
        require(self.evaluator.model.config.hidden_size == expected, 'Model does not match the declared policy configuration')
        return self.evaluator

    def loader(self, task):
        return self.manifests[task], image_loader(self.args.data_root, task)

    def save(self, key, value):
        require('/' not in key and key not in self.records, 'Invalid or duplicate result identifier')
        write_json(self.output / 'records' / (key + '.json'), value)
        self.records.append(key)

    def finish(self, details):
        result = {'records': len(self.records), 'runtime_sha256': self.runtime_sha,
                  'episode_manifest_sha256': self.bundle_sha, **details}
        write_json(self.output / 'completion.json', result)
        return result
