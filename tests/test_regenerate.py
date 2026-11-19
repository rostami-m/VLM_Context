"""CPU checks of result reduction, uncertainty direction, and archive parity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from vlm_context.io import read_json, source_path
from vlm_context.regenerate import aggregate_q05, verify_q05_metrics, _interval, core_seed_rows, regenerate


def record(image, gold, strict, summed, **flags):
    predictions = {'strict': strict, 'sum': summed, 'mean': summed, 'historical': strict}
    return dict(task='synthetic', arm='frozen', condition='standard', k=1,
        episode_seed=0, fit_seed=0, image_id=image, label_space=['red', 'blue'], gold=gold,
        predictions=predictions, invalid={key: value is None for key, value in predictions.items()},
        correct={key: int(value == gold) for key, value in predictions.items()},
        ambiguous=False, natural_label_emission=False, option_echo=flags.get('option_echo', False),
        parser_disagreement=False, scorer_disagreement=False)


class ReductionTests(unittest.TestCase):
    def test_invalid_generation_is_retained_without_affecting_likelihood_interface(self):
        rows = [record('a', 0, None, 0, option_echo=True), record('b', 1, 1, 1)]
        cells = {row['scorer']: row for row in aggregate_q05(rows)}
        strict, summed = cells['strict'], cells['sum']
        self.assertEqual(strict['n'], 2)
        self.assertEqual(strict['accuracy'], .5)
        self.assertEqual(strict['invalid_count'], 1)
        self.assertEqual(strict['confusion_with_invalid'], [[0, 0, 1], [0, 1, 0]])
        self.assertEqual(strict['option_echo_count'], 1)
        self.assertEqual(summed['accuracy'], 1)
        self.assertEqual(summed['invalid_count'], 0)

    def test_repeated_contexts_stay_in_distinct_cells_and_duplicates_fail(self):
        a = record('a', 0, 0, 0)
        b = dict(a, episode_seed=1, fit_seed=1)
        self.assertEqual(len(aggregate_q05([a, b])), 8)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_q05([a, a])

    def test_recorded_flags_and_archived_confusions_must_match(self):
        row = record('a', 0, 0, 0)
        broken = copy.deepcopy(row)
        broken['correct']['sum'] = 0
        with self.assertRaisesRegex(ValueError, 'correctness'):
            aggregate_q05([broken])
        cells = aggregate_q05([row])
        archived = copy.deepcopy(cells)
        archived[0]['confusion_with_invalid'][0][0] = 0
        with self.assertRaisesRegex(ValueError, 'confusion'):
            verify_q05_metrics(cells, archived)

    def test_comparator_minus_reference_interval_reverses_both_sign_and_order(self):
        self.assertEqual(_interval([-.4, -.2], reverse=True), [20, 40])
        self.assertEqual(_interval([-.4, .2], reverse=True), [-20, 40])
        with self.assertRaisesRegex(ValueError, 'interval'):
            _interval([.4, .2])

    def test_core_seed_traces_must_reproduce_saved_mean(self):
        cell = dict(arm='joint', task='x', condition='standard', k=0, reference_accuracy=.5)
        seeds = dict(arm='joint', task='x', condition='standard', k=0,
            metrics={'reference_accuracy': {'paired_replicate_estimates':
                [{'seed': 0, 'estimate': 0}, {'seed': 1, 'estimate': .5}, {'seed': 2, 'estimate': 1}]}})
        self.assertEqual([r['accuracy_pct'] for r in core_seed_rows([cell], [seeds])], [0, 50, 100])
        with self.assertRaisesRegex(ValueError, 'mean disagrees'):
            core_seed_rows([dict(cell, reference_accuracy=.6)], [seeds])

    def test_paper_archive_reconstruction_and_unrounded_transfer_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'results'
            manifest = regenerate(output, figures=False)
            saved = read_json(source_path('data/paper/q05_metrics.json.gz'))
            recomputed = json.loads((output / 'tables/q05_per_cell_diagnostics.json').read_text())
            self.assertEqual(len(recomputed), len(saved))
            self.assertTrue(verify_q05_metrics(recomputed, saved)['all_fields_match'])
            core = read_json(source_path('data/paper/core_cells.json'))
            transfer = json.loads((output / 'tables/core_transfer.json').read_text())
            row = next(r for r in transfer if r['task'] == 'fashion4' and r['adapter'] == 'Joint')
            original = next(r for r in core if r['task'] == 'fashion4' and
                r['arm'] == 'joint_uncapped_r16_episode_remap' and r['condition'] == 'standard' and r['k'] == 16)
            self.assertEqual(row['standard_K16_pct'], 100 * original['reference_accuracy'])
            self.assertEqual(manifest['tables']['core_transfer'], len(transfer))
            self.assertFalse(manifest['figures_generated'])
            self.assertIn('data/paper/q05_per_image.json.gz', manifest['inputs'])
            with self.assertRaisesRegex(ValueError, 'fresh'):
                regenerate(output, figures=False)


if __name__ == '__main__':
    unittest.main()
