"""Controller execution with zero-shot retained-task matrices and phase restoration."""
from ..continual.contract import (KS,VERSION,Trigger,controller_training_rows,retain_replay,write,require,digest,sha)
from ..continual.policies import accuracy_matrix_metrics
from ..continual.native import source_hashes

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
    self.change_precision(False)
    from vlm_context.adaptation import parameter_hashes
    after = {n: h for n, h in parameter_hashes(self.evaluator.model, self.torch).items() if '.lora_' not in n}
    require(after == self.base_hash, 'frozen base changed at completion')
    result = {'version': VERSION, 'experiment': self.config['experiment'], 'stream_id': self.stream['stream_id'], 'arm': self.config['arm'], 'mode': self.config['mode'], 'task_order': self.stream['task_order'][:n], 'matrices': matrices, 'metrics': {s: accuracy_matrix_metrics(m) for s, m in matrices.items()}, 'stages': summaries, 'controller_events': events, 'optimizer_steps': self.updates, 'records': self.records, 'costs': self.costs, 'verification_passed': True, 'scope': 'fixed image panel; repeated queries/seeds are not independent images', 'base_unchanged': True}
    write(self.output / 'result.json', result)
    artifacts = {str(p.relative_to(self.output)): sha(p) for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != 'progress.json'}
    write(self.output / 'validation.json', {'verification_passed': True, 'config_sha256': digest(self.config), 'source_sha256': source_hashes(), 'artifact_sha256': artifacts, 'record_count': self.records, 'optimizer_steps': self.updates, 'population_claim': False})
    self.progress('complete', n)
