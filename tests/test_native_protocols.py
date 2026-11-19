"""CPU checks for native answer boundaries and frozen open-task protocols."""
import unittest
from types import SimpleNamespace
from vlm_context.io import read_json,source_path
from vlm_context.native_families import boundary,strict_parse,historical_parse
from vlm_context.open_tasks import vqa_score,legacy_image_id,render_rule
from vlm_context.experiments.vqa import scheduled,summarize

class NativeProtocols(unittest.TestCase):
    def test_complete_label_plus_native_terminator(self):
        b=boundary([11],[11,12,13,14],'red fox',lambda ids:'red fox',{13})
        self.assertEqual(b['label_token_ids'],[12])
        self.assertEqual(b['scored_token_ids'],[12,13])
        self.assertEqual(b['trailing_template_token_ids'],[14])
        for prefix,full in [([11],[10,12,13]),([11],[11,13]),([11],[11,12])]:
            with self.assertRaises(ValueError):boundary(prefix,full,'red fox',lambda ids:'red fox',{13})
        with self.assertRaises(ValueError):boundary([11],[11,12,13],'red fox',lambda ids:'red',{13})

    def test_strict_and_historical_parsers_are_distinct(self):
        labels=['car','airplane']
        self.assertIsNone(strict_parse('an airplane',labels))
        self.assertEqual(historical_parse('an airplane',labels),1)
        self.assertEqual(strict_parse(' airplane\n',labels),1)

    def test_official_vqa_soft_credit(self):
        self.assertAlmostEqual(vqa_score('yes',['yes']*3+['no']*7),.9)
        self.assertAlmostEqual(vqa_score('Two.',['2']*10),1)
        self.assertAlmostEqual(vqa_score('a dog',['dog']*10),1)

    def test_frozen_open_task_panel_counts(self):
        for task,count in [('vqav2',80),('textvqa',100),('scienceqa',100),('rule',320)]:
            _,rows=scheduled({'task':task},SimpleNamespace(task=None,seed=0,smoke=False))
            self.assertEqual(len(rows),count)
            self.assertTrue(all(e['partition']=='confirmatory' for e in rows))

    def test_rule_metric_requires_both_counterfactuals(self):
        _,rows=scheduled({'task':'rule'},SimpleNamespace(task=None,seed=0,smoke=False))
        result=summarize([{'episode':e,'score':float(e['rule']=='left')} for e in rows],'rule')
        self.assertEqual(len(result['cells']),4)
        self.assertTrue(all(c['pairs']==40 and c['paired_both_rules_accuracy']==0 for c in result['cells']))
        with self.assertRaises(ValueError):summarize([{'episode':rows[0],'score':1}], 'rule')

    def test_rule_blueprints_recreate_saved_pixels(self):
        try:import PIL
        except ImportError:self.skipTest('Pillow not installed')
        j=read_json(source_path('data/protocols/q11/rule_scenes_development.json'))
        for scene in list(j['scenes'].values())[:8]:
            self.assertEqual(legacy_image_id(render_rule(scene,j['config'])),scene['image_id'])

    def test_replication_mixed_training_schedule(self):
        j=read_json(source_path('data/protocols/replication/training.json.gz'))
        for seed in [0,1,2]:
            rows=[e for e in j if e['fit_seed']==seed]
            self.assertEqual(len(rows),150)
            self.assertEqual(sum(e['condition']=='standard' for e in rows),75)
            self.assertEqual(sum(e['condition']=='remap' for e in rows),75)
        for name in ['replication_qwen25','replication_internvl','native_idefics2','native_mantis','native_openflamingo','q11_vqav2','q11_textvqa','q11_scienceqa','q11_rule']:
            c=read_json(source_path('configs/'+name+'.json'))
            self.assertIn(c['family'],['replication','native_probes','vqa','different_rule'])


class NativeCheckpointLineage(unittest.TestCase):
    def make_fit(self,root,*,kind='replication',smoke=False):
        import json
        from pathlib import Path
        from vlm_context.io import sha256
        self.spec={'model_id':'example/native','revision':'pinned-revision'}
        checkpoint=Path(root)/'checkpoint';checkpoint.mkdir()
        (checkpoint/'adapter_config.json').write_text('{}')
        (checkpoint/'adapter_model.safetensors').write_bytes(b'fixture adapter bytes')
        run={'study':kind,'mode':'train','task':'shapes4','arm':'connector' if kind=='replication' else 'joint_uncapped_r16_episode_remap','seed':1,'model_id':self.spec['model_id'],'model_revision':self.spec['revision'],'smoke_only':smoke,'episodes':2 if smoke else(150 if kind=='replication' else 250)}
        runtime={'model_id':self.spec['model_id'],'revision':self.spec['revision']} if kind=='replication' else {'runtime':{'model_id':self.spec['model_id'],'model_revision':self.spec['revision']}}
        recipe={'arm':'connector','seed':1,'steps':run['episodes'],'smoke':smoke,'rank':8,'alpha':16,'dropout':0,'weight_decay':0,'learning_rate':1e-4,'gradient_clip':1}
        values={'run.json':run,'runtime.json':runtime,'recipe.json':recipe,'reload.json':{'exact_parameter_match':True},'base_unchanged.json':{'before':{},'after':{}}}
        for name,v in values.items():(Path(root)/name).write_text(json.dumps(v))
        receipt={'completed_steps':run['episodes'],'frozen_base_unchanged':True,'reload_verified':True,'trainable_parameters':12800000,'checkpoint_files':{p.name:sha256(p)for p in checkpoint.iterdir()},'evidence_files':{name:sha256(Path(root)/name)for name in values}}
        (Path(root)/'fit_complete.json').write_text(json.dumps(receipt))
        return checkpoint

    def test_frozen_and_training_reject_adapter(self):
        from vlm_context.native_checkpoints import validate_replication,validate_q11
        for fn in [validate_replication,validate_q11]:
            with self.assertRaises(ValueError):fn('unneeded',arm='frozen',seed=0,spec={})
        with self.assertRaises(ValueError):validate_replication(None,arm='frozen',seed=0,spec={},training=True)
        with self.assertRaises(ValueError):validate_replication('unneeded',arm='connector',seed=0,spec={},training=True)

    def test_q13_rejects_wrong_lineage_and_changed_checkpoint(self):
        import tempfile
        from vlm_context.native_checkpoints import validate_replication
        with tempfile.TemporaryDirectory()as root:
            p=self.make_fit(root)
            validate_replication(p,arm='connector',seed=1,spec=self.spec)
            for arm,seed,spec in [('late',1,self.spec),('connector',2,self.spec),('connector',1,{**self.spec,'revision':'wrong'})]:
                with self.assertRaises(ValueError):validate_replication(p,arm=arm,seed=seed,spec=spec)
            (p/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_replication(p,arm='connector',seed=1,spec=self.spec)

    def test_q13_requires_complete_fit_and_bound_metadata(self):
        import tempfile,json
        from pathlib import Path
        from vlm_context.native_checkpoints import validate_replication
        with tempfile.TemporaryDirectory()as root:
            p=self.make_fit(root);receipt=Path(root)/'fit_complete.json';value=json.loads(receipt.read_text())
            receipt.write_text(json.dumps({**value,'completed_steps':149}))
            with self.assertRaises(ValueError):validate_replication(p,arm='connector',seed=1,spec=self.spec)
            receipt.write_text(json.dumps(value));recipe=Path(root)/'recipe.json';recipe.write_text(recipe.read_text()+' ')
            with self.assertRaises(ValueError):validate_replication(p,arm='connector',seed=1,spec=self.spec)

    def test_smoke_adapter_requires_explicit_smoke_evaluation(self):
        import tempfile
        from vlm_context.native_checkpoints import validate_replication,validate_q11
        for kind,fn,arm in [('replication',validate_replication,'connector'),('core',validate_q11,'joint')]:
            with tempfile.TemporaryDirectory()as root:
                p=self.make_fit(root,kind=kind,smoke=True)
                with self.assertRaises(ValueError):fn(p,arm=arm,seed=1,spec=self.spec)
                fn(p,arm=arm,seed=1,spec=self.spec,smoke=True)

    def test_q11_requires_matching_shapes_joint_fit(self):
        import tempfile,json
        from pathlib import Path
        from vlm_context.native_checkpoints import validate_q11
        with tempfile.TemporaryDirectory()as root:
            p=self.make_fit(root,kind='core')
            validate_q11(p,arm='joint',seed=1,spec=self.spec)
            with self.assertRaises(ValueError):validate_q11(p,arm='joint',seed=0,spec=self.spec)
            with self.assertRaises(ValueError):validate_q11(p,arm='joint',seed=1,spec={**self.spec,'model_id':'another/model'})
            r=Path(root)/'run.json';v=json.loads(r.read_text());r.write_text(json.dumps({**v,'arm':'late_parameter_matched_r34_35_episode_remap'}))
            with self.assertRaises(ValueError):validate_q11(p,arm='joint',seed=1,spec=self.spec)

if __name__=='__main__':unittest.main()
