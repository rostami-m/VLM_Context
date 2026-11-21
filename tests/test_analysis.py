import numpy as np
import unittest
from vlm_context.analysis_statistics import weighted_bc_predictions,bc_draws,paired_family
from vlm_context.analysis_q11 import collect


class StatisticalAnalysisTests(unittest.TestCase):
    def test_bc_refits_on_each_resample_without_gold_in_fit(self):
        logits=[[-1.,-3.,-8.,-9.],[-5.,-1.,-8.,-9.]]
        weights=np.array([[2.,0.],[0.,2.],[1.,1.]])
        gold=[0,1]
        expected=[]
        for w in weights:
            p=weighted_bc_predictions(logits,[int(v)for v in w]);expected.append(sum(n*(a==b)for n,a,b in zip(w,p,gold))/sum(w))
        np.testing.assert_allclose(bc_draws(logits,weights,gold),expected)
        fixed=weighted_bc_predictions(logits,[1,1])
        fixed_draws=[sum(n*(a==b)for n,a,b in zip(w,fixed,gold))/sum(w)for w in weights]
        assert expected!=fixed_draws  # Fixed calibration inside bootstrap is a different estimator.


    def test_paired_family_retains_image_cluster_and_rejects_missing_context(self):
        base={(f'{c}-{i}',s):{'class_id':c,'value':0.}for c in range(4)for i in range(20)for s in range(3)}
        improved={k:{**v,'value':.25}for k,v in base.items()}
        r=paired_family({'a':base,'b':improved},'a',fitted=True,draws=30)
        assert r['methods']['b']['paired_simultaneous_family_band95']==[.25,.25]
        assert r['methods']['b']['paired_fitted_seed_interval']['t95_df2']==[.25,.25]
        improved.pop(next(iter(improved)))
        with self.assertRaisesRegex(ValueError,'coverage'):paired_family({'a':base,'b':improved},'a',draws=2)


    def test_q11_rule_requires_both_sides_and_keeps_image_identity(self):
        def row(side,pred):
            return {'method':'m','task':'rule','condition':'implicit_natural','k':16,'image_id':'scene','seed':0,'rule':side,'candidates':[{'label':'red','scored_token_ids':[2,3],'native_end_token':3,'token_log_probabilities':[-1.,-.1],'sum_logp':-1.1}], 'sum_prediction':'red','answers':[pred],'score':float(pred=='red')}
        a=row('left','red');b=row('right','blue')
        assert collect([a,b])[('m','rule','implicit_natural',16)][('scene',0)]==0
        with self.assertRaisesRegex(ValueError,'Incomplete'):collect([a])

    def test_seed_reference_reuse_requires_complete_explicit_baseline(self):
        from vlm_context.analysis_q11 import seed_summary
        left={(q,s):1. for q in ('a','b')for s in range(3)}
        baseline={(q,0):0. for q in ('a','b')}
        with self.assertRaisesRegex(ValueError,'Reference'):
            seed_summary(left,baseline)
        result=seed_summary(left,baseline,reuse=True)
        self.assertEqual(result['t_interval_95'],[1.,1.])
        mixed={**baseline,('a',1):.5}
        with self.assertRaisesRegex(ValueError,'Reference'):
            seed_summary(left,mixed,reuse=True)
