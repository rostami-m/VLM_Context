"""CPU checks of scientific invariants; no model download or accelerator required."""
import math
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vlm_context.io import read_json, source_path, write_json, sha256
from vlm_context.protocol import ImageRecord, prepare_episode, validate_splits, parse_diagnostics, sequence_likelihoods, summarize_predictions
from vlm_context.training_primitives import complete_answer_mask, parameter_hashes
from vlm_context.adaptation import diagnostic_ledger
from vlm_context.data_contract import dataset_roles
from vlm_context.explicit import expand_variant
from vlm_context.experiments.classification import scheduled, validate_checkpoint
from vlm_context.calibration import contextual_calibration, batch_calibration, SCORE_DOMAIN
from vlm_context.kernels.position import expand, validate_triplet
from vlm_context.kernels.position_schedule import position_triplet
from vlm_context.baseline_scoring import neutral_episode, neutral_loader

class CoreProtocols(unittest.TestCase):
    def test_full_answer_eos_causal_shift(self):
        mask = complete_answer_mask([4, 9, 11], [7, 8], 2)
        self.assertEqual(mask['labels'], [-100, -100, -100, 7, 8, 2])
        self.assertEqual(mask['prediction_positions'], [2, 3, 4])
        self.assertEqual(mask['answer_token_ids'], [7, 8, 2])
        with self.assertRaises(ValueError): complete_answer_mask([4], [2], 2)

    def test_sum_mean_can_disagree_and_ties_are_explicit(self):
        result = sequence_likelihoods([[-.4, -.4, -.4], [-.6, -.5], [-2.], [-3.]])
        self.assertEqual(result['sum_prediction'], 1)
        self.assertEqual(result['mean_prediction'], 0)
        self.assertEqual(sequence_likelihoods([[-1.], [-1.]])['sum_prediction'], 0)
        for invalid in ([[math.nan]], [[.2]], [[]]):
            with self.assertRaises(ValueError): sequence_likelihoods(invalid)

    def test_strict_parser_preserves_invalid_denominator(self):
        labels = ['cat', 'dog', 'ankle boot', 'bird']
        self.assertEqual(parse_diagnostics('Ankle   Boot.', labels)['prediction'], 2)
        for text in ['not a cat', 'cat or dog', 'cat dog ankle boot bird']:
            self.assertIsNone(parse_diagnostics(text, labels)['prediction'])
        metrics = summarize_predictions([{'gold': 0, 'prediction': 0}, {'gold': 1, 'prediction': None}], 4)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(metrics['invalid_count'], 1)
        self.assertEqual(metrics['confusion'][1][-1], 1)

    def test_decoded_content_leakage_is_rejected(self):
        a = ImageRecord('a', '0' * 64, 0)
        b = ImageRecord('different-id', '0' * 64, 1)
        with self.assertRaises(ValueError): validate_splits({'train': [a], 'test': [b]})

    def test_rng_streams_pair_supports_across_label_conditions(self):
        query = ImageRecord('query', 'f' * 64, 2)
        pool = [ImageRecord(str(i), f'{i:064x}', i % 4) for i in range(32)]
        natural = prepare_episode(query, pool, ['a', 'b', 'c', 'd'], 8, 2)
        remap = prepare_episode(query, pool, ['a', 'b', 'c', 'd'], 8, 2, remap_labels=['dax','wug','blicket','koba'])
        self.assertEqual(natural['supports'], remap['supports'])
        self.assertEqual(natural['option_order'], remap['option_order'])
        self.assertEqual(len({s['image_id'] for s in remap['supports']}), 8)
        self.assertEqual([sum(s['class_id'] == c for s in remap['supports']) for c in range(4)], [2]*4)

    def test_all_manifest_roles_are_disjoint(self):
        for task in ['shapes4','cifar4','fashion4']:
            manifest = read_json(source_path(f'data/protocols/datasets/{task}_manifest.json.gz'))
            roles = dataset_roles(task, manifest)
            self.assertEqual(len(roles['confirmation_query']), 80)
            self.assertEqual(len(roles['adapter_training_support']), 256)

    def test_target_budgets_and_omissions(self):
        self.assertEqual(diagnostic_ledger('connector')['trainable_parameters'], 135168)
        for locus in ['early','middle','late','matched_all_depth']:
            self.assertEqual(diagnostic_ledger(locus)['trainable_parameters'], 122880)
        recipes = read_json(source_path('data/protocols/core/target_recipes.json'))['recipes']
        self.assertEqual(recipes['joint_uncapped_r16']['trainable_parameters'], 12800000)
        self.assertEqual(recipes['late_parameter_matched_r34_35']['trainable_parameters'], 12800000)
        windows = [r for n,r in recipes.items() if n.startswith('window_')]
        self.assertEqual(len(windows), 7)
        self.assertEqual({r['trainable_parameters'] for r in windows}, {1318912})
        self.assertFalse(any(r['name'].startswith('visual.') for r in recipes['joint_without_connector_r16']['modules']))

    def test_training_lengths_read_frozen_schedules(self):
        for study,arm,expected in [('q05','connector_episode_remap',150),('core','joint_uncapped_r16_natural',250),('depth','window_0_3_r8',150)]:
            args = SimpleNamespace(seed=0,task='shapes4',arm=arm,mode='train',smoke=False)
            _,_,episodes,_,_ = scheduled(read_json(source_path(f'configs/{study}.json')),args)
            self.assertEqual(len(episodes), expected)
            self.assertTrue(all(e['query']['image_id'] not in [s['image_id'] for s in e['supports']] for e in episodes))

    def test_classification_smoke_covers_zero_and_long_context(self):
        args = SimpleNamespace(seed=0, task='shapes4', arm='frozen', mode='evaluate', smoke=True)
        _, _, episodes, _, _ = scheduled(read_json(source_path('configs/q05.json')), args)
        self.assertEqual([(e['k'], e['condition']) for e in episodes], [(0, 'standard'), (16, 'episode_remap')])

    def test_checkpoint_identity_and_smoke_are_not_interchangeable(self):
        from vlm_context.qwen import MODEL_ID, MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            (checkpoint / 'adapter.bin').write_bytes(b'fixture-weights')
            run = {'mode':'train','task':'shapes4','arm':'connector_episode_remap','seed':0,
                   'smoke_only':False,'episodes':150}
            fit = {'completed_steps':150,'checkpoint_files':{'adapter.bin':sha256(checkpoint/'adapter.bin')}}
            write_json(root/'run.json',run);write_json(root/'fit_complete.json',fit)
            write_json(root/'runtime.json',{'runtime':{'model_id':MODEL_ID,'model_revision':MODEL_REVISION}})
            args = SimpleNamespace(checkpoint=checkpoint,seed=0,smoke=False)
            validate_checkpoint({'study':'q05'},args,'shapes4','connector_episode_remap')
            args.seed=1
            with self.assertRaises(ValueError):validate_checkpoint({'study':'q05'},args,'shapes4','connector_episode_remap')
            args.seed=0
            with self.assertRaises(ValueError):validate_checkpoint({'study':'q05'},args,'shapes4','frozen')
            run.update(smoke_only=True,episodes=2);fit['completed_steps']=2
            write_json(root/'run.json',run,overwrite=True);write_json(root/'fit_complete.json',fit,overwrite=True)
            with self.assertRaises(ValueError):validate_checkpoint({'study':'q05'},args,'shapes4','connector_episode_remap')
            args.smoke=True
            validate_checkpoint({'study':'q05'},args,'shapes4','connector_episode_remap')
            (checkpoint/'adapter.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_checkpoint({'study':'q05'},args,'shapes4','connector_episode_remap')

    def test_position_triplets_change_order_only(self):
        manifest = read_json(source_path('data/protocols/datasets/shapes4_manifest.json.gz'))
        triplets = read_json(source_path('data/protocols/interventions/position_triplets.json.gz'))
        self.assertEqual(len(expand(triplets, manifest, 'confirmation_query')), 1440)
        for t in triplets:
            validate_triplet(t, manifest, 'confirmation_query')
            regenerated = position_triplet(manifest['images'],manifest['roles']['evaluation_support'],t['query_id'],t['seed'],t['condition'])
            self.assertEqual(json.loads(json.dumps(regenerated)),t)

    def test_calibration_domain_and_unlabeled_batch(self):
        context='0'*64; labels=['a','b']
        cc=contextual_calibration([-2.,-3.],[[-1.,-4.]], label_order=labels,content_free_label_order=labels,
            query_context=context,content_free_contexts=[context],score_domain=SCORE_DOMAIN)
        self.assertEqual(cc['prediction'],1)
        bc=batch_calibration([[-1.,-4.],[-3.,-2.]], label_order=labels,context_hashes=[context]*2,
            score_domain=SCORE_DOMAIN,batch_role='unlabeled_deployment')
        self.assertEqual(bc['bias_log_scores'],[-2.,-3.])
        self.assertTrue(bc['transductive'])
        with self.assertRaises(ValueError):
            batch_calibration([[-1.,-4.],[-3.,-2.]],label_order=labels,context_hashes=[context,'1'*64],
                score_domain=SCORE_DOMAIN,batch_role='unlabeled_deployment')

    def test_neutral_image_hash_matches_pixels(self):
        try: import PIL
        except ImportError: self.skipTest('Pillow not installed')
        from vlm_context.shapes import decoded_hash
        ep=neutral_episode({'query': {'image_id':'original'}})
        im=neutral_loader(lambda _: None)(ep['query']['image_id'])
        self.assertEqual(decoded_hash(im),ep['query']['content_sha256'])

    def test_immutable_evidence_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json'
            write_json(path,{'correct':True});write_json(path,{'correct':True})
            with self.assertRaises(ValueError):write_json(path,{'correct':False})

    def test_parameter_hash_tracks_tensor_values(self):
        try: import torch
        except ImportError: self.skipTest('Optional torch not installed')
        model=torch.nn.Linear(2,1,bias=False)
        before=parameter_hashes(model,torch)
        with torch.no_grad():model.weight[0,0]+=1
        self.assertNotEqual(before,parameter_hashes(model,torch))

if __name__=='__main__':unittest.main()
