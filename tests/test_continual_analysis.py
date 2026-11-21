"""CPU tests for image identity, zero-shot retention and controller work accounting."""
from copy import deepcopy
import unittest

from vlm_context.continual_analysis import collect, bootstrap, predictions, _controller_events, PREFIXES


def retention_records():
    records = []
    for stage in range(5):
        for task in range(stage+1):
            for query in range(80):
                gold = query//20
                predicted = (gold+1)%4 if stage == 4 and task == 0 else gold
                records.append(dict(arm='joint', stream_id='fixture', task_order=list(range(5)),
                                    stage=stage, task=task, kind='matrix', image_id=f'{task}/{query:02d}',
                                    gold=gold, k=0, labels=['A', 'B', 'C', 'D'],
                                    greedy_answer=['A', 'B', 'C', 'D'][predicted],
                                    candidate_token_ids=[[i+32, 99] for i in range(4)],
                                    token_logps=[[-.1 if i == predicted else -2., -.1] for i in range(4)]))
    return records


class ContinualAnalysisTests(unittest.TestCase):
    def test_retention_uses_paired_diagonal_and_final_stages(self):
        metadata = [dict(arm='joint', stream_id='fixture', task_order=list(range(5)))]
        stream = collect(retention_records(), metadata)['joint', 'fixture']
        self.assertEqual(stream['retention']['sum'], {'AA': .8, 'BWT': -.25})
        self.assertEqual(stream['matrices']['sum'][0], [1, None, None, None, None])
        self.assertEqual(stream['matrices']['sum'][4], [0, 1, 1, 1, 1])

    def test_rejects_duplicate_missing_nonzero_shot_and_changed_identity(self):
        metadata = [dict(arm='joint', stream_id='fixture', task_order=list(range(5)))]
        for change in ('duplicate', 'missing', 'context', 'class'):
            rows = retention_records()
            if change == 'duplicate': rows.append(deepcopy(rows[0]))
            elif change == 'missing': rows.pop()
            elif change == 'context': rows[-1]['k'] = 1
            else: rows[-1]['gold'] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                collect(rows, metadata)

    def test_bootstrap_keeps_twenty_task_class_strata(self):
        values = {f'{task}/{cls}/{i}': (task, cls, float(task == 0))
                  for task in range(5) for cls in range(4) for i in range(20)}
        result = bootstrap(values, draws=100)
        self.assertEqual(result['estimate'], .2)
        self.assertEqual(result['descriptive_95_interval'], [.2, .2])
        values.pop(next(iter(values)))
        with self.assertRaisesRegex(ValueError, '400 images'):
            bootstrap(values, draws=100)

    def test_invalid_generated_answer_kept_separate_from_likelihood(self):
        row = retention_records()[0]
        row['greedy_answer'] = 'A or B'
        result = predictions(row)
        self.assertEqual(result['sum'], 0)
        self.assertIsNone(result['strict'])
        self.assertEqual(result['legacy'], 0)

    def test_controller_counts_charged_and_cached_probe_work_separately(self):
        events, online = [], {}
        for stage in range(5):
            for prefix in PREFIXES:
                event = dict(stage_index=stage, task=stage, available_prefix=prefix,
                             consolidation=False, gate_before=dict(selected_k=0, charged_queries=40,
                                                                  cached_queries_reused=80))
                events.append(event)
                online[stage, prefix] = {'image': {'k': 0}}
        events[-1].update(consolidation=True, training_steps=80,
                          gate_after=dict(selected_k=1, charged_queries=80, cached_queries_reused=0))
        online[4, 16]['image']['k'] = 1
        stream = dict(meta=dict(events=events, task_order=list(range(5))), online=online)
        result = _controller_events(stream)
        self.assertEqual(result, dict(optimizer_steps=80, consolidation_events=1,
                                      charged_probe_queries=1280, cached_probe_queries_reused=2400))
        online[4, 16]['image']['k'] = 0
        with self.assertRaisesRegex(ValueError, 'selected context'):
            _controller_events(stream)


if __name__ == '__main__':
    unittest.main()
