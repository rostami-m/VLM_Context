"""Failure-oriented CPU tests for saved native output reductions."""
from copy import deepcopy
import unittest

from vlm_context.native_analysis import reduce_records, score_record


def record(seed=0, image='image0', text='Bird.'):
    return dict(method='frozen', task='cifar4', condition='standard', k=0,
                image_id=image, seed=seed, gold=0, label_space=['bird', 'cat'],
                displayed_options=['cat', 'bird'], greedy_text=text,
                recorded_exact_format_prediction_index=None,
                recorded_sum_prediction='bird', recorded_mean_prediction='bird',
                candidates=[dict(label=label, label_token_ids=[token], scored_token_ids=[token, 99],
                                 native_end_token=99, token_log_probabilities=[logp, -.5],
                                 sum_logp=logp-.5, mean_logp=(logp-.5)/2)
                            for label, token, logp in [('cat', 10, -2.), ('bird', 11, -1.)]])


class NativeAnalysisTests(unittest.TestCase):
    def test_predictions_use_displayed_option_order_and_preserve_score(self):
        result = score_record(record())
        self.assertEqual(result, dict(sum=0, mean=0, strict=0, exact_format=None, sum_position=1))

    def test_contexts_are_clustered_by_image_and_invalids_retained(self):
        rows = [record(seed, image, 'Bird.' if image == 'a' else 'The answer is bird.')
                for seed in range(3) for image in ('a', 'b')]
        result = reduce_records(rows, {'frozen': {'arm': 'frozen', 'model_id': 'model'}}, draws=100)
        cell = result['cells']['frozen/cifar4/standard/0']
        self.assertEqual(cell['accuracy']['images'], 2)
        self.assertEqual(cell['records'], 6)
        self.assertEqual(cell['accuracy']['mean'], 1)
        self.assertEqual(cell['strict_invalid_rate'], .5)
        self.assertEqual(cell['strict_generation_accuracy'], .5)
        self.assertEqual(cell['exact_format_invalid_rate'], 1)
        self.assertEqual(cell['normalized_format_recoveries'], 3)
        self.assertEqual(cell['strict_confusion_with_invalid_column'][0], [3, 0, 3])

    def test_rejects_duplicate_and_missing_contexts(self):
        methods = {'frozen': {'arm': 'frozen', 'model_id': 'model'}}
        rows = [record(seed) for seed in range(3)]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            reduce_records(rows + [rows[0]], methods, draws=100)
        with self.assertRaisesRegex(ValueError, 'all three'):
            reduce_records(rows[:2], methods, draws=100)

    def test_rejects_changed_scores_terminator_and_old_exact_field(self):
        for change in ('sum', 'eos', 'exact', 'argmax', 'nonfinite'):
            row = deepcopy(record())
            if change == 'sum': row['candidates'][0]['sum_logp'] += 1
            elif change == 'eos': row['candidates'][0]['scored_token_ids'].pop()
            elif change == 'exact': row['recorded_exact_format_prediction_index'] = 1
            elif change == 'argmax': row['recorded_sum_prediction'] = 'cat'
            else: row['candidates'][0]['token_log_probabilities'][0] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_record(row)

    def test_rejects_unpaired_checkpoint_comparison(self):
        rows = [record(seed) for seed in range(3)]
        adapted = [{**record(seed, 'different'), 'method': 'late'} for seed in range(3)]
        methods = {'frozen': {'arm': 'frozen', 'model_id': 'model'},
                   'late': {'arm': 'late', 'model_id': 'model'}}
        with self.assertRaisesRegex(ValueError, 'Unpaired native checkpoint'):
            reduce_records(rows + adapted, methods, draws=100)

    def test_rejects_checkpoint_from_another_fitting_seed(self):
        row = {**record(), 'checkpoint_descriptor_sha256': 'wrong'}
        methods = {'frozen': {'arm': 'frozen', 'model_id': 'model', 'checkpoint_descriptors': {}}}
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            reduce_records([row], methods, draws=100)


if __name__ == '__main__':
    unittest.main()
