"""Persistent LoRA streams with complete-answer supervision and saved native traces.

The scientific loop is extracted from the accepted experiment implementation.
Q17 retains BF16 base/inference; the controller subclass implements Q18's
value-preserving phase changes. Output folders are fresh per stream.
"""
from __future__ import annotations
import gc,json,os,time
from pathlib import Path
from .contract import (KS,VERSION,Trigger,controller_training_rows,digest,episode,policy_spec,require,retain_replay,sha,write)
from .policies import CopyGate,accuracy_matrix_metrics

def source_hashes():
    here=Path(__file__).resolve().parent
    return {p.name:sha(p) for p in sorted(here.glob('*.py'))}

def attach(evaluator, expected):
    from peft import LoraConfig, get_peft_model
    torch = evaluator.torch
    actual = {n: (m.in_features, m.out_features) for n, m in evaluator.model.named_modules() if isinstance(m, torch.nn.Linear)}
    for row in expected['modules']:
        require(actual.get(row['name']) == (row['in_features'], row['out_features']), 'native target dimension mismatch')
    rank = min(expected['rank_pattern'].values())
    adapter = get_peft_model(evaluator.model, LoraConfig(r=rank, lora_alpha=2 * rank,
        lora_dropout=.05, target_modules=expected['target_regex'], bias='none',
        rank_pattern=expected['rank_pattern'], alpha_pattern=expected['alpha_pattern']))
    evaluator.model = adapter.get_base_model()
    injected = {n: m for n, m in evaluator.model.named_modules() if hasattr(m, 'lora_A')}
    require(set(injected) == set(expected['rank_pattern']), 'native adapter target set mismatch')
    for name, module in injected.items():
        require(module.lora_A['default'].weight.shape == (expected['rank_pattern'][name], actual[name][0]), 'actual adapter rank mismatch')
        require(bool((module.lora_B['default'].weight == 0).all().item()), 'nonzero initial effective LoRA update')
    rows = [{'name': n, 'shape': list(t.shape), 'numel': t.numel(), 'dtype': str(t.dtype)} for n, t in adapter.named_parameters() if t.requires_grad]
    require(all('.lora_' in r['name'] for r in rows) and sum(r['numel'] for r in rows) == 12800000, 'unmatched trainable parameter budget')
    adapter.eval()
    return adapter, rows


class NativeRun:
    def __init__(self, args, config, data, schedule, targets):
        from vlm_context.qwen import QwenEvaluator, resolve_runtime
        self.args, self.config, self.data, self.targets = args, config, data, targets
        self.output = args.output.resolve()
        require(not self.output.exists(), 'fresh output directory required; preserve partial attempts')
        self.stream = next(s for s in schedule['streams'] if s['stream_id'] == config['stream_id'])
        self.seed = self.stream['initialization_seed']
        self.smoke = config['mode'] == 'smoke'
        self.controller = config['experiment'] == 'Q18'
        self.development = self.smoke or config['mode'] == 'development'
        self.stages = self.stream['stages'][:2] if self.smoke else self.stream['stages']
        self.runtime = resolve_runtime(config['snapshot'], config['device'], load_dtype='bfloat16')
        self.evaluator = QwenEvaluator(self.runtime, seed=self.seed)
        self.torch = self.evaluator.torch
        self.adapter = None
        self.records, self.updates, self.costs = 0, 0, {'probe_seconds': 0., 'prediction_seconds': 0., 'training_seconds': 0., 'reload_seconds': 0.}
        write(self.output / 'run_manifest.json', {'config': config, 'config_sha256': digest(config), 'sources': source_hashes(), 'runtime': self.runtime, 'policy': policy_spec() if self.controller else None})
        first = self.data['tasks'][str(self.stages[0]['current_task'])]['roles']['development_query'][0]
        base = self.evaluate(first, (), verify=True)
        self.adapter, ledger = attach(self.evaluator, targets['recipes']['joint' if self.controller else targets['arms'][config['arm']]])
        zero = self.evaluate(first, (), verify=True)
        from vlm_context.adaptation import compare_roundtrip, parameter_hashes
        compare_roundtrip(base, zero, .05)
        write(self.output / 'initialization.json', {'base_record': base, 'zero_adapter_record': zero, 'zero_update_score_parity_passed': True, 'trainable_tensors': ledger})
        self.base_hash = {n: h for n, h in parameter_hashes(self.evaluator.model, self.torch).items() if '.lora_' not in n}
        write(self.output / 'frozen_base_before.json', self.base_hash)
        self.probe_cache = {}

    def loader(self, identity):
        from PIL import Image
        from .materialize import decoded_hash
        row = self.data['images'][identity]
        path = Path(self.config['data']) / row['path']
        require(sha(path) == row['file_sha256'], 'image-file drift')
        with Image.open(path) as f:
            result = f.convert('RGB')
        require(result.size == (112, 112) and decoded_hash(result) == row['content_sha256'], 'decoded-image drift')
        return result

    def evaluate(self, identity, supports=(), verify=False):
        e = episode(self.data, identity, supports, self.seed)
        r = self.evaluator.evaluate(e, self.loader, list('ABCD'), verify=verify)
        if verify:
            from vlm_context.protocol import sequence_likelihoods
            v = r['independent_validation']
            require(v and r['scores']['sum_prediction'] == sequence_likelihoods(v['independent_stepwise_token_logps'])['sum_prediction'], 'stepwise argmax mismatch')
        self.records += 1
        return r

    def progress(self, phase, stage, **extra):
        value = {'at_epoch': time.time(), 'phase': phase, 'stage': stage, 'stages': len(self.stages), 'records': self.records, 'optimizer_steps': self.updates, **extra}
        tmp = self.output / 'progress.tmp'
        tmp.write_text(json.dumps(value, allow_nan=False) + '\n')
        os.replace(tmp, self.output / 'progress.json')
        print(json.dumps(value), flush=True)

    def train(self, stage_index, rows, allowed):
        from vlm_context.adaptation import training_batch, require_positive_norms
        require(all(r['query_id'] in allowed for r in rows), 'training reads unreleased/discarded data')
        torch, model = self.torch, self.evaluator.model
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads(); model.config.use_cache = False; self.adapter.train()
        named = [(n, t) for n, t in self.adapter.named_parameters() if t.requires_grad]
        # Reset optimizer state at task boundaries; adapter weights persist.
        optimizer = torch.optim.AdamW([t for _, t in named], lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01)
        self.evaluator.synchronize(); started = time.monotonic()
        for index, row in enumerate(rows):
            optimizer.zero_grad(set_to_none=True)
            e = episode(self.data, row['query_id'], (), self.seed)
            require(e['supervised_label'] == row['output_label'], 'training label drift')
            batch, metadata = training_batch(self.evaluator, e, self.loader)
            result = self.adapter(**batch, use_cache=False, return_dict=True)
            loss = result.loss
            require(bool(torch.isfinite(loss).item()), 'nonfinite training loss')
            target = torch.tensor(metadata['supervision']['answer_token_ids'], device=self.evaluator.device)
            logits = result.logits[0, metadata['supervision']['prediction_positions']].detach().float()
            manual = torch.nn.functional.cross_entropy(logits, target)
            difference = abs(float(loss.item()) - float(manual.item()))
            require(difference <= .005, 'whole-answer plus EOS loss mismatch')
            loss.backward()
            gradients, before = [], {}
            for name, tensor in named:
                require(tensor.grad is not None and bool(torch.isfinite(tensor.grad).all().item()), 'missing/nonfinite gradient')
                gradients.append({'name': name, 'l2_norm': float(tensor.grad.float().norm().item())})
                before[name] = tensor.detach().float().clone()
            require_positive_norms([x['l2_norm'] for x in gradients], 'gradient norms')
            optimizer.step(); updates = []
            for name, tensor in named:
                require(bool(torch.isfinite(tensor).all().item()) and all(not torch.is_tensor(v) or bool(torch.isfinite(v).all().item()) for v in optimizer.state[tensor].values()), 'nonfinite parameter/optimizer')
                updates.append({'name': name, 'l2_norm': float((tensor.detach().float() - before[name]).norm().item())})
            require_positive_norms([x['l2_norm'] for x in updates], 'update norms')
            write(self.output / f'stages/{stage_index}/steps/{index:04d}.json', {'training_row': row, 'loss': float(loss.item()), 'manual_loss': float(manual.item()), 'loss_absolute_difference': difference, 'gradient_norms': gradients, 'update_norms': updates, **metadata})
            self.updates += 1
            self.progress('training', stage_index, step=index + 1, steps=len(rows))
            del batch, result, loss, manual, logits, before
        self.evaluator.synchronize(); self.costs['training_seconds'] += time.monotonic() - started
        del optimizer
        model.gradient_checkpointing_disable(); model.disable_input_require_grads()
        model.config.use_cache = True; self.adapter.eval()
        self.probe_cache.clear()

    def checkpoint(self, stage_index, current_task):
        from peft import PeftModel
        from vlm_context.adaptation import compare_roundtrip, parameter_hashes
        from vlm_context.qwen import QwenEvaluator
        before = parameter_hashes(self.evaluator.model, self.torch)
        require({n: h for n, h in before.items() if '.lora_' not in n} == self.base_hash, 'frozen base changed')
        folder = self.output / f'stages/{stage_index}/checkpoint'
        self.adapter.save_pretrained(folder, safe_serialization=True)
        probes = self.data['tasks'][str(current_task)]['roles']['development_query'][:2]
        reference = [self.evaluate(i, verify=True) for i in probes]
        write(self.output / f'stages/{stage_index}/pre_reload.json', {'parameter_sha256': before, 'records': reference})
        started = time.monotonic()
        self.adapter = None; self.evaluator = None; gc.collect()
        if self.config['device'] == 'mps':
            self.torch.mps.empty_cache()
        self.evaluator = QwenEvaluator(self.runtime, seed=self.seed)
        self.adapter = PeftModel.from_pretrained(self.evaluator.model, folder, is_trainable=True)
        self.evaluator.model = self.adapter.get_base_model(); self.adapter.eval()
        after = parameter_hashes(self.evaluator.model, self.torch)
        require(before == after, 'adapter/base roundtrip hash mismatch')
        records = [self.evaluate(i, verify=True) for i in probes]
        for a, b in zip(reference, records):
            compare_roundtrip(a, b, .05)
        self.evaluator.synchronize(); elapsed = time.monotonic() - started
        self.costs['reload_seconds'] += elapsed
        write(self.output / f'stages/{stage_index}/reload_validation.json', {'verification_passed': True, 'exact_parameter_hash_roundtrip': True, 'records': records, 'seconds': elapsed, 'checkpoint_sha256': {p.name: sha(p) for p in folder.iterdir() if p.is_file()}})

    def evaluate_cell(self, stage_index, task, role, supports=(), category='matrix', prefix=None):
        ids = self.data['tasks'][str(task)]['roles'][role]
        ids = ids[:4] if self.smoke else ids
        correct = {'sum': 0, 'mean': 0, 'strict': 0, 'legacy': 0}
        for index, identity in enumerate(ids):
            r = self.evaluate(identity, supports, verify=index == 0)
            gold = self.data['images'][identity]['class_id']
            for name, prediction in [('sum', r['scores']['sum_prediction']), ('mean', r['scores']['mean_prediction']), ('strict', r['parser']['prediction']), ('legacy', r['parser']['legacy_prediction'])]:
                correct[name] += prediction == gold
            sub = category if prefix is None else f'{category}/prefix_{prefix}'
            write(self.output / f'stages/{stage_index}/{sub}/task_{task}/{index:04d}.json', r)
            self.costs['prediction_seconds'] += r['seconds']
            self.progress(category, stage_index, task=task, query=index + 1, queries=len(ids))
        return {'task': task, 'role': role, 'queries': len(ids), 'k': len(supports), 'correct': correct, 'accuracy': {k: v / len(ids) for k, v in correct.items()}}

    def gate(self, stage_index, block, maximum_k, threshold, phase):
        gate = CopyGate(threshold, probe_count=40)
        charged, reused = 0, 0
        for k in KS[1:]:
            if k > maximum_k or gate.stopped:
                break
            supports = block['nested_support_ids'][str(k)]
            key = (stage_index, k, self.updates)
            if key not in self.probe_cache:
                rows = []
                for index, identity in enumerate(block['controller_development_query_ids']):
                    r = self.evaluate(identity, supports, verify=index == 0)
                    write(self.output / f'stages/{stage_index}/gate/{phase}/k_{k}/{index:04d}.json', r)
                    rows.append(r)
                    self.costs['probe_seconds'] += r['seconds']; charged += 1
                # Gate consumes predictions and support endpoint only; no query gold.
                endpoint = self.data['images'][supports[-1]]['class_id']
                count = sum(r['parser']['prediction'] == endpoint for r in rows)
                self.probe_cache[key] = {'copied': count, 'total': len(rows)}
            else:
                reused += 40
            count = self.probe_cache[key]
            gate.observe(k, count['copied'], count['total'])
        return {'selected_k': gate.selected_k, 'threshold': threshold, 'observations': gate.observations, 'charged_queries': charged, 'cached_queries_reused': reused, 'maximum_available_k': maximum_k}

    def execute(self):
        n = len(self.stages); matrices = {score: [[None] * n for _ in range(n)] for score in ('sum', 'mean', 'strict', 'legacy')}
        summaries, events, controller_memory, selected_by_task = [], [], [], {}
        threshold = self.config.get('gate_threshold')
        for index, stage in enumerate(self.stages):
            task = stage['current_task']; self.progress('task_begin', index, task=task)
            if not self.controller:
                replay = self.config['arm'] == 'late_replay'
                rows = stage['replay_reference_training_rows' if replay else 'training_rows']
                allowed = set(stage['released_current_training_ids']) | (set(stage['replay_buffer_before']) if replay else set())
                smoke_rows = [rows[0], rows[4]] if replay and index else rows[:2]
                self.train(index, smoke_rows if self.smoke else rows, allowed)
                self.checkpoint(index, task)
            else:
                trigger = Trigger(self.config['arm']); block = stage['controller_support']
                for available in KS:
                    gate = self.gate(index, block, available, threshold, f'before_prefix_{available}')
                    fired = trigger.observe(available, gate['selected_k'])
                    event = {'stage_index': index, 'task': task, 'available_prefix': available, 'gate_before': gate, 'consolidation': fired, 'retained_replay_before': list(controller_memory)}
                    if fired:
                        current = block['nested_support_ids'][str(available)]
                        rows = controller_training_rows(current, controller_memory, self.data['images'], task, self.seed, 2 if self.smoke else 80)
                        self.train(index, rows, set(current) | set(controller_memory))
                        self.checkpoint(index, task)
                        event['training_ids'] = sorted({r['query_id'] for r in rows})
                        event['training_steps'] = len(rows)
                        gate = self.gate(index, block, available, threshold, f'after_prefix_{available}')
                        event['gate_after'] = gate
                    supports = block['nested_support_ids'][str(gate['selected_k'])]
                    role = 'development_query' if self.development else 'controller_final'
                    event['online_accuracy'] = self.evaluate_cell(index, task, role, supports, 'online', available)
                    selected_by_task[task] = list(supports)
                    events.append(event)
                    write(self.output / f'stages/{index}/decisions/prefix_{available}.json', event)
                # Every policy has the same available labels and memory ceiling.
                controller_memory = retain_replay(controller_memory, block['nested_support_ids']['16'], self.data['images'], self.seed)
                write(self.output / f'stages/{index}/memory.json', {'retained_ids': controller_memory, 'maximum_images': 16})
                if not trigger.updated:
                    self.checkpoint(index, task)
            cells = []
            for column, previous in enumerate(self.stream['task_order'][:index + 1]):
                role = 'development_query' if self.development else ('controller_final' if self.controller else 'confirmation_query')
                # Reusable current-task context is allowed only for online accuracy.
                # Retention matrix is zero-shot for all policies and CL arms.
                cell = self.evaluate_cell(index, previous, role)
                cells.append(cell)
                for score in matrices:
                    matrices[score][index][column] = cell['accuracy'][score]
            summaries.append({'stage_index': index, 'current_task': task, 'cells': cells})
            write(self.output / f'stages/{index}/summary.json', summaries[-1])
        from vlm_context.adaptation import parameter_hashes
        after = {n: h for n, h in parameter_hashes(self.evaluator.model, self.torch).items() if '.lora_' not in n}
        require(after == self.base_hash, 'frozen base changed at completion')
        result = {'version': VERSION, 'experiment': self.config['experiment'], 'stream_id': self.stream['stream_id'], 'arm': self.config['arm'], 'mode': self.config['mode'], 'task_order': self.stream['task_order'][:n], 'matrices': matrices, 'metrics': {s: accuracy_matrix_metrics(m) for s, m in matrices.items()}, 'stages': summaries, 'controller_events': events, 'optimizer_steps': self.updates, 'records': self.records, 'costs': self.costs, 'verification_passed': True, 'scope': 'fixed image panel; repeated queries/seeds are not independent images', 'base_unchanged': True}
        write(self.output / 'result.json', result)
        artifacts = {str(p.relative_to(self.output)): sha(p) for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != 'progress.json'}
        write(self.output / 'validation.json', {'verification_passed': True, 'config_sha256': digest(self.config), 'source_sha256': source_hashes(), 'artifact_sha256': artifacts, 'record_count': self.records, 'optimizer_steps': self.updates, 'population_claim': False})
        self.progress('complete', n)
