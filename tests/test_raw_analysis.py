"""CPU invariants for analysis from generated text and token-level scores."""
import copy
import unittest

from vlm_context.io import read_json, source_path
from vlm_context.q05_analysis import reduce_record, verify_q05_outcomes
from vlm_context.position_analysis import analyze_position, verify_position


class RawAnalysisTests(unittest.TestCase):
    def row(self):
        return dict(task='example', arm='frozen', condition='standard', k=0,
            episode_seed=0, fit_seed=0, image_id='q', gold=1,
            label_space=['bird', 'cat'], natural_labels=['bird', 'cat'],
            greedy_answer='Cat.', eos_token_id=99,
            candidate_tokens=[dict(label='bird', label_token_ids=[1], scored_token_ids=[1, 99]),
                              dict(label='cat', label_token_ids=[2, 3], scored_token_ids=[2, 3, 99])],
            per_token_label_log_probabilities=[[-1., -1.], [-.9, -.9, -.9]])

    def test_full_token_sum_and_mean_remain_distinct_from_normalized_generation(self):
        result = reduce_record(self.row())
        self.assertEqual(result['predictions'], dict(sum=0, mean=1, strict=1, historical=1))
        self.assertTrue(result['scorer_disagreement'])
        self.assertEqual(result['correct']['sum'], 0)
        self.assertEqual(result['correct']['strict'], 1)

    def test_prose_is_not_promoted_to_strict_answer(self):
        row = self.row()
        row['greedy_answer'] = 'The answer is cat.'
        result = reduce_record(row)
        self.assertIsNone(result['predictions']['strict'])
        self.assertEqual(result['predictions']['historical'], 1)
        self.assertTrue(result['parser_disagreement'])
        self.assertTrue(result['invalid']['strict'])

    def test_image_level_outcome_verification_rejects_changed_pairing(self):
        result = reduce_record(self.row())
        self.assertTrue(verify_q05_outcomes([result], [result])['all_per_image_outcomes_match'])
        with self.assertRaisesRegex(ValueError, 'coverage'):
            verify_q05_outcomes([result], [dict(result, image_id='other')])
        with self.assertRaisesRegex(ValueError, 'outcome'):
            verify_q05_outcomes([result], [dict(result, gold=0)])

    def test_missing_or_duplicate_end_token_and_token_alignment_fail(self):
        for broken in ('missing', 'duplicate', 'alignment'):
            row = self.row()
            if broken == 'missing':
                row['candidate_tokens'][0]['scored_token_ids'].pop()
            elif broken == 'duplicate':
                row['candidate_tokens'][0]['label_token_ids'].append(99)
                row['candidate_tokens'][0]['scored_token_ids'].append(99)
            else:
                row['per_token_label_log_probabilities'][0].pop()
            with self.assertRaises(ValueError):
                reduce_record(row)

    def test_all_position_cell_and_paired_means_match_archive_from_raw_scores(self):
        rows = read_json(source_path('data/analysis/position_raw.json.gz'))
        result = analyze_position(rows)
        check = verify_position(result, read_json(source_path('data/paper/position_results.json')))
        self.assertEqual(check['records'], 1440)
        self.assertEqual(check['cells'], 54)
        self.assertTrue(check['all_values_match'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analyze_position(rows + [copy.deepcopy(rows[0])])


if __name__ == '__main__':
    unittest.main()
