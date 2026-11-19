import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from vlm_context.io import read_json, source_path, write_json, sha256
from vlm_context.kernels.gate.policy import first_crossing, select_threshold, eligibility, mean_difference
from vlm_context.kernels.continual.contract import (
    Trigger, controller_training_rows, retain_replay, validate_data, episode)
from vlm_context.kernels.inject.operations import choose_readers, READERS
from vlm_context.kernels.inject.readers import completed_reader


class GateTests(unittest.TestCase):
    def test_first_crossing_is_strict_and_stops_probing(self):
        # Ten matches among forty is accepted at .25; the next larger prefix fails.
        counts = {'1': {'copied': 10, 'total': 40}, '2': {'copied': 11, 'total': 40}}
        result = first_crossing(counts, .25)
        self.assertEqual((result['selected_k'], result['probe_calls'], result['first_crossing_k']), (1, 80, 2))

    def test_invalid_outputs_do_not_shrink_denominator(self):
        with self.assertRaises(ValueError):
            first_crossing({'1': {'copied': 10, 'total': 30}}, .25)

    def test_tied_accuracy_prefers_smaller_selected_prefix(self):
        block = {'copy_counts': {str(k): {'copied': 12, 'total': 40} for k in (1, 2, 4, 8, 16)},
                 'development_accuracy': {str(k): .5 for k in (0, 1, 2, 4, 8, 16)}}
        self.assertEqual(select_threshold([block])['threshold'], .25)

    def test_eligibility_boundaries(self):
        self.assertTrue(eligibility([True]*20+[False]*20, [True]*24+[False]*16)['eligible'])
        self.assertFalse(eligibility([True]*21+[False]*19, [True]*24+[False]*16)['eligible'])
        self.assertFalse(eligibility([True]*19+[False]*21, [True]*23+[False]*17)['eligible'])

    def test_paired_vector_difference(self):
        self.assertEqual(mean_difference([([4., 9.], [1., 4.]), ([8., 3.], [3., 6.])]), [4., 1.])
        with self.assertRaises(ValueError):
            mean_difference([([float('nan')], [0.])])


class ContinualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_json(source_path('data/protocols/continual/image_manifest.json.gz'))
        cls.schedule = read_json(source_path('data/protocols/continual/stream_schedule.json.gz'))
        cls.targets = read_json(source_path('data/protocols/continual/continual_target_contract.json'))

    def test_exact_frozen_protocol(self):
        result = validate_data(self.manifest, self.schedule, self.targets)
        self.assertEqual(result['images'], 2360)
        self.assertEqual(result['persistent_replay_limit'], 16)

    def test_future_data_is_rejected(self):
        schedule = copy.deepcopy(self.schedule)
        stage = schedule['streams'][0]['stages'][0]
        future = next(i for i, r in self.manifest['images'].items() if r['role'] == 'continual_training'
                      and r['task_id'] != stage['current_task'])
        stage['training_rows'][0]['query_id'] = future
        with self.assertRaises(ValueError):
            validate_data(self.manifest, schedule, self.targets)

    def test_training_receives_only_prefix_and_retained_replay(self):
        stage = self.schedule['streams'][0]['stages'][1]
        prefix = stage['controller_support']['nested_support_ids']['8']
        replay = stage['replay_buffer_before']
        rows = controller_training_rows(prefix, replay, self.manifest['images'], stage['current_task'], 0)
        self.assertEqual(sum(r['example_origin'] == 'replay' for r in rows), 16)
        self.assertTrue({r['query_id'] for r in rows} <= set(prefix + replay))
        retained = retain_replay(replay, prefix, self.manifest['images'], 0)
        self.assertLessEqual(len(retained), 16)
        self.assertTrue(set(retained) <= set(prefix + replay))

    def test_fixed_labels_and_no_task_id_in_prompt_episode(self):
        stage = self.schedule['streams'][0]['stages'][0]
        supports = stage['controller_support']['nested_support_ids']['8']
        query = next(i for i in stage['released_current_training_ids'] if i not in supports)
        value = episode(self.manifest, query, supports)
        self.assertEqual(value['label_space'], list('ABCD'))
        self.assertNotIn('task_id', value['query'])
        self.assertTrue(all('task_id' not in s for s in value['supports']))
        with self.assertRaises(ValueError): episode(self.manifest, query, supports + [query])

    def test_controller_requires_consecutive_underuse_and_only_one_update(self):
        trigger = Trigger('gate_triggered_consolidation')
        self.assertFalse(trigger.observe(1, 0))
        self.assertFalse(trigger.observe(2, 2))  # Reset the streak.
        self.assertFalse(trigger.observe(4, 0))
        self.assertTrue(trigger.observe(8, 0))
        self.assertFalse(trigger.observe(16, 0))
        fixed = Trigger('fixed_schedule_consolidation')
        self.assertEqual([fixed.observe(k, 0) for k in (1, 2, 4, 8, 16)], [False, False, False, True, False])
        prompt = Trigger('prompt_only_gate')
        self.assertFalse(any(prompt.observe(k, 0) for k in (1, 2, 4, 8, 16)))


class InjectTests(unittest.TestCase):
    def artifacts(self, directory):
        paths = []
        for reader in READERS:
            path = Path(directory) / (reader + '.json')
            write_json(path, {'kind': 'inject-reader-eligibility-v1', 'reader': reader,
                'final_queries_used': False, 'bundle_sha256': 'same-data', 'runtime_sha256': 'same-model',
                'blocks': [{'task': t, 'seed': seed, 'supported_correct': 24, 'zero_correct': 20,
                            'reader_identity': {'trainable_parameters': 0 if reader == 'frozen' else 100},
                            'selection_seconds': 1.} for t in ('shapes4', 'cifar4') for seed in range(3)]})
            paths.append(path)
        return paths

    def test_reader_selection_tie_uses_fewer_parameters(self):
        with TemporaryDirectory() as directory:
            selected = choose_readers(Path(directory), self.artifacts(directory), None)
            self.assertEqual(selected['reader_by_task'], {'shapes4': 'frozen', 'cifar4': 'frozen'})

    def test_confirmation_cannot_select_reader(self):
        with TemporaryDirectory() as directory:
            paths = self.artifacts(directory)
            artifact = read_json(paths[0]); artifact['final_queries_used'] = True
            write_json(paths[0], artifact, overwrite=True)
            with self.assertRaises(ValueError): choose_readers(Path(directory), paths, None)

    def test_incomplete_reader_seed_grid_is_rejected(self):
        with TemporaryDirectory() as directory:
            paths = self.artifacts(directory)
            artifact = read_json(paths[0]); artifact['blocks'].pop()
            write_json(paths[0], artifact, overwrite=True)
            with self.assertRaises(ValueError): choose_readers(Path(directory), paths, None)

    def test_only_complete_source_fits_are_accepted(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            (checkpoint/'adapter.safetensors').write_bytes(b'fixture')
            write_json(root/'fit_complete.json', {'completed_steps': 150, 'trainable_parameters': 135168,
                'checkpoint_files': {'adapter.safetensors': sha256(checkpoint/'adapter.safetensors')}})
            run = {'study': 'q05', 'mode': 'train', 'task': 'shapes4', 'arm': 'connector_episode_remap',
                   'seed': 0, 'smoke_only': False, 'episodes': 150}
            write_json(root/'run.json', run)
            session = SimpleNamespace(config={'reader_checkpoints': {'connector_episode_remap': str(checkpoint)}})
            completed_reader(session, 'connector_episode_remap', 0)
            run['smoke_only'] = True; write_json(root/'run.json', run, overwrite=True)
            with self.assertRaises(ValueError): completed_reader(session, 'connector_episode_remap', 0)


if __name__ == '__main__': unittest.main()
