"""Exact explicit-support KV reuse with complete multimodal RoPE positions."""
from __future__ import annotations
import copy,time
from ..gate.common import require,validate_score

class SupportCache:
    def __init__(self,evaluator,inputs,metadata,k):
        self.ev,self.k=evaluator,k
        evaluator.synchronize();started=time.monotonic()
        require(k>0,'cached support reference requires demonstrations')
        t=evaluator.torch;ids=metadata['input_token_ids'];starts=[]
        token=evaluator.model.config.image_token_id
        for i,v in enumerate(ids):
            if v==token and (i==0 or ids[i-1]!=token):starts.append(i)
        require(len(starts)==k+1,'image count differs from requested prefix')
        self.cut=starts[-1]-1
        require(ids[self.cut]==evaluator.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),'query vision boundary missing')
        self.prefix_ids=ids[:self.cut]
        self.grid=inputs['image_grid_thw'][:k].clone()
        self.pixel_rows=sum(int(t.prod(grid).item()) for grid in self.grid)
        self.pixels=inputs['pixel_values'][:self.pixel_rows].clone()
        positions,rope=evaluator.model.get_rope_index(inputs['input_ids'],inputs['image_grid_thw'],None,inputs['attention_mask'])
        prefix={'input_ids':inputs['input_ids'][:,:self.cut],
                'attention_mask':inputs['attention_mask'][:,:self.cut],
                'pixel_values':self.pixels,'image_grid_thw':self.grid,
                'position_ids':positions[:,:,:self.cut],
                'cache_position':t.arange(self.cut,device=evaluator.device)}
        with t.inference_mode():
            result=evaluator.model(**prefix,use_cache=True,return_dict=True)
        evaluator.synchronize();self.seconds=time.monotonic()-started
        self.cache=result.past_key_values
        self.prefix_tokens=self.cut
        del result

    def score(self,inputs,metadata,episode):
        ev,t=self.ev,self.ev.torch
        ev.synchronize();start=time.monotonic()
        require(metadata['input_token_ids'][:self.cut]==self.prefix_ids,'support prefix token bytes changed')
        require(t.equal(inputs['image_grid_thw'][:self.k],self.grid) and
                t.equal(inputs['pixel_values'][:self.pixel_rows],self.pixels),'support processor tensors changed')
        n=inputs['input_ids'].shape[1]
        positions,rope=ev.model.get_rope_index(inputs['input_ids'],inputs['image_grid_thw'],None,inputs['attention_mask'])
        tail={'input_ids':inputs['input_ids'][:,self.cut:],'attention_mask':inputs['attention_mask'],
              'pixel_values':inputs['pixel_values'][self.pixel_rows:],'image_grid_thw':inputs['image_grid_thw'][self.k:],
              'position_ids':positions[:,:,self.cut:],
              'cache_position':t.arange(self.cut,n,device=ev.device),
              'past_key_values':copy.deepcopy(self.cache)}
        with t.inference_mode():
            ev.model.rope_deltas=rope.clone()
            out=ev.model(**tail,use_cache=True,return_dict=True)
            first=out.logits[0,-1].float().log_softmax(-1);base=out.past_key_values
            del out
            rows=[]
            for candidate in metadata['candidate_tokens']:
                ev.model.rope_deltas=rope.clone();cache=copy.deepcopy(base);next_logp=first;row=[]
                for index,token in enumerate(candidate['scored_token_ids']):
                    row.append(float(next_logp[token].item()))
                    if index+1<len(candidate['scored_token_ids']):
                        next_logp,cache=ev._next(token,cache,n+index)
                rows.append(row);del cache
            ev.model.rope_deltas=rope.clone();cache=copy.deepcopy(base);next_logp=first;tokens=[];logps=[]
            for index in range(ev.max_new_tokens):
                token=int(next_logp.argmax().item());tokens.append(token);logps.append(float(next_logp[token].item()))
                if token in ev.stop_ids:break
                next_logp,cache=ev._next(token,cache,n+index)
            del cache,base
        ev.synchronize()
        from vlm_context.protocol import sequence_likelihoods as likelihoods
        from vlm_context.protocol import parse_diagnostics
        text=ev.processor.tokenizer.decode(tokens,skip_special_tokens=True)
        result={'status':'ok','episode':episode,**metadata,'per_token_label_log_probabilities':rows,'scores':likelihoods(rows),
                'greedy_token_ids':tokens,'greedy_token_log_probabilities':logps,'greedy_text':text,
                'parse':parse_diagnostics(text,episode['label_space']),'seconds':time.monotonic()-start,
                'cached_support_reference':{'prefix_tokens_cached':self.cut,'query_suffix_tokens':n-self.cut,
                                            'support_cache_setup_seconds':self.seconds,'cache_charged_once_per_block':True}}
        validate_score(result,episode)
        return result
