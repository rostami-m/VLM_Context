"""Explicit support-label rendering for held-out mappings and corruption controls."""
from __future__ import annotations
from copy import deepcopy
import math, time
from .io import require
from .protocol import parse_diagnostics, sequence_likelihoods
from .qwen_base import QwenEvaluator


def render_explicit(episode,image_loader):
    supports=episode['supports']; labels=episode.get('support_labels')
    require(isinstance(labels,list) and len(labels)==len(supports),'explicit labels required')
    require(all(s in episode['label_space'] for s in labels),'support label outside candidates')
    content=[]; images=[]
    for support,label in zip(supports,labels):
        images.append(image_loader(support['image_id']))
        content.extend([{'type':'image'},{'type':'text','text':'This image shows: '+label+'.'}])
    images.append(image_loader(episode['query']['image_id']))
    content.extend([{'type':'image'},{'type':'text','text':'Classify the last image. Answer with exactly one of ['+
        ', '.join(episode['displayed_options'])+'] and nothing else.'}])
    return [{'role':'user','content':content}],images



def expand_variant(context,variant):
    require(sorted(variant['support_order'])==list(range(context['k'])),'invalid support order')
    e=deepcopy(context)
    e['supports']=[e['supports'][i] for i in variant['support_order']]
    for key in ['label_space','option_order','displayed_options','support_labels','condition',
                'target_label','reference_target_label','corruption','consistent_class_mapping']:
        e[key]=deepcopy(variant[key])
    require(len(e['support_labels'])==e['k'],'explicit label length differs')
    return e



def validate_three_scoring_paths(cached,full,stepwise,candidates,tolerance):
    require(len(candidates)==len(cached)==len(full)==len(stepwise)==4,
        'all three scorers must preserve four candidate spans')
    for candidate,*rows in zip(candidates,cached,full,stepwise):
        expected=len(candidate['scored_token_ids'])
        require(expected>0 and all(len(row)==expected for row in rows),
            'cached/full/stepwise candidate token span differs')
        require(all(type(v) in (int,float) and math.isfinite(v) and v<=0 for row in rows for v in row),
            'scoring paths require finite nonpositive token log probabilities')
    scores=[sequence_likelihoods(rows) for rows in [cached,full,stepwise]]
    require(len({s['sum_prediction'] for s in scores})==1,
        'cached/full/stepwise primary argmax differs')
    errors=[abs(a-b) for ca,fu,st in zip(cached,full,stepwise) for a,b in list(zip(ca,fu))+list(zip(ca,st))]
    require(errors and max(errors)<=tolerance,'explicit cached/full/stepwise mismatch')
    return max(errors)



class ExplicitEvaluator(QwenEvaluator):
    def evaluate(self,episode,image_loader,natural_labels,verify=False):
        """Reuse frozen scoring machinery; only the renderer handles corruption."""
        self.synchronize(); started=time.monotonic()
        messages,images=render_explicit(episode,image_loader)
        inputs,metadata=self.encode(messages,images,episode['label_space'])
        logps,generated,generated_logps=self._cached_candidates(inputs,metadata)
        scores=sequence_likelihoods(logps)
        raw=self.processor.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        parsing=parse_diagnostics(raw,episode['label_space'],natural_labels=natural_labels)
        validation=None
        if verify:
            fulls=[self.independent_full_forward(inputs,c['scored_token_ids']) for c in metadata['candidate_tokens']]
            steps=[self.independent_stepwise(inputs,c['scored_token_ids']) for c in metadata['candidate_tokens']]
            tolerance=.0001 if self.device=='cpu' else .05
            error=validate_three_scoring_paths(logps,fulls,steps,metadata['candidate_tokens'],tolerance)
            with self.torch.inference_mode():
                native=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.max_new_tokens)
            require(native[0,inputs['input_ids'].shape[1]:].tolist()==generated,'explicit/native greedy mismatch')
            validation={'full_forward_token_logps':fulls,'independent_stepwise_token_logps':steps,
                'maximum_absolute_logp_difference':error,'absolute_tolerance':tolerance,
                'generation_matches_transformers_generate':True}
        self.synchronize()
        return {'status':'ok','episode':episode,**metadata,'scores':scores,
            'per_token_label_log_probabilities':logps,'greedy_token_ids':generated,
            'greedy_token_log_probabilities':generated_logps,'greedy_answer':raw,'parser':parsing,
            'ties':{name:[i for i,x in enumerate(scores[name]) if x==max(scores[name])] for name in ['sum','mean']},
            'generation_reached_limit':len(generated)==self.max_new_tokens and generated[-1] not in self.stop_ids,
            'historical_strict_disagreement':parsing['legacy_prediction']!=parsing['prediction'],
            'sum_mean_disagreement':scores['sum_prediction']!=scores['mean_prediction'],
            'independent_validation':validation,'seconds':time.monotonic()-started}
