"""Exact native position edits and attention interventions for Qwen2-VL."""
from __future__ import annotations
import copy,math
from ..gate.policy import require

def classification_renderer(episode,loader):
    from vlm_context.cama import classification_messages
    rendered=copy.deepcopy(episode)
    for support in rendered['supports']:
        support['class_id']=support.get('display_class_id',support['class_id'])
    return classification_messages(rendered,loader)


def encode(evaluator,episode,loader):
    messages,images=classification_renderer(episode,loader)
    return evaluator.encode(messages,images,episode['label_space'])


def segmentation(evaluator,metadata,episode):
    from vlm_context.cama import classification_segments
    ids=metadata['input_token_ids'];tokenizer=evaluator.processor.tokenizer
    segments=classification_segments(ids,tokenizer,evaluator.model.config.image_token_id)
    require(len(segments)==len(episode['supports'])+1,'image count mismatch')
    labels=[]
    for segment,support in zip(segments,episode['supports']):
        positions=segment['text'];tokens=[ids[i] for i in positions]
        text=tokenizer.decode(tokens,skip_special_tokens=False)
        label=episode['label_space'][support.get('display_class_id',support['class_id'])]
        left=text.index(label);right=left+len(label)
        if getattr(tokenizer,'is_fast',False):
            encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            offsets=encoded['offset_mapping']
        else:
            pieces=[tokenizer.decode([token],skip_special_tokens=False) for token in tokens]
            require(''.join(pieces)==text,'slow tokenizer does not preserve exact label character boundaries')
            offsets=[];cursor=0
            for piece in pieces:
                offsets.append((cursor,cursor+len(piece)));cursor+=len(piece)
            encoded=tokenizer(text,add_special_tokens=False)
        require(list(encoded['input_ids'])==tokens,'support label segmentation retokenized')
        selected=[positions[i] for i,(a,b) in enumerate(offsets) if a<right and b>left]
        require(bool(selected),'empty label token span')
        labels.append(selected)
    query=segments[-1]['image']+segments[-1]['text']
    count=sum(map(len,labels))
    image_control=[]
    for segment,span in zip(segments,labels):
        require(len(segment['image'])>=len(span),'image control cannot match label token count')
        image_control.extend(segment['image'][:len(span)])
    first_image=segments[0]['image'][0]
    text_control=[i for i in range(1,first_image-1) if ids[i] not in tokenizer.all_special_ids][:count]
    require(len(text_control)==count,'insufficient distinct instruction-text control columns')
    return {'segments':segments,'label_columns':[v for row in labels for v in row],
            'image_control_columns':image_control,'text_control_columns':text_control,
            'query_rows':query,'reference_position':len(ids)-1,
            'unrelated_patch_position':text_control[0]}


def aligned(clean,shuffled,clean_segments,shuffled_segments):
    require(len(clean['input_token_ids'])==len(shuffled['input_token_ids']),'clean/shuffled token lengths differ')
    require(clean['image_grid_thw']==shuffled['image_grid_thw'] and
            clean['candidate_tokens']==shuffled['candidate_tokens'],'image/candidate alignment changed')
    require(clean_segments==shuffled_segments,'token spans differ across label shuffling')
    labels=set(clean_segments['label_columns'])
    changed={i for i,(a,b) in enumerate(zip(clean['input_token_ids'],shuffled['input_token_ids'])) if a!=b}
    require(bool(changed) and changed<=labels,'corruption changed tokens outside displayed support labels')
    return {'prefix_length':len(clean['input_token_ids']),'changed_positions':sorted(changed),
            'unchanged_outside_support_labels':True,'segments':clean_segments}


class HiddenEdit:
    def __init__(self,evaluator,layer,prefix_length,position,vector,mode='replace'):
        self.ev,self.layer,self.n,self.position=evaluator,layer,prefix_length,position
        self.vector,self.mode=vector,mode
        require(mode in ('replace','add','noop') and 0<=position<prefix_length,'invalid residual edit')
        self.calls=0

    def __enter__(self):
        self.handle=self.ev.model.model.layers[self.layer].register_forward_hook(self.hook)
        return self

    def hook(self,module,args,output):
        hidden=output[0]
        if hidden.shape[1]<self.n:
            return None
        self.calls+=1
        if self.mode=='noop':
            return None
        v=self.ev.torch.as_tensor(self.vector,device=hidden.device,dtype=hidden.dtype)
        require(list(v.shape)==[hidden.shape[-1]] and bool(self.ev.torch.isfinite(v).all()),'invalid residual vector')
        changed=hidden.clone()
        if self.mode=='replace':
            changed[0,self.position]=v
        else:
            changed[0,self.position]=changed[0,self.position]+v
        return (changed,*output[1:])

    def __exit__(self,*args):
        self.handle.remove()


class AttentionCut:
    def __init__(self,evaluator,layer,prefix_length,rows,columns,heads=None):
        self.ev,self.layer,self.n=evaluator,layer,prefix_length
        self.rows,self.columns,self.heads=list(rows),list(columns),heads
        require(len(set(rows))==len(rows) and len(set(columns))==len(columns),'duplicate intervention cells')
        require(not columns or max(columns)<min(rows),'noncausal attention cut')
        self.calls=0

    def __enter__(self):
        self.handle=self.ev.model.model.layers[self.layer].self_attn.register_forward_pre_hook(self.hook,with_kwargs=True)
        return self

    def hook(self,module,args,kwargs):
        require(not args,'unsupported attention signature')
        length=kwargs['hidden_states'].shape[1]
        if length<self.n:
            return None
        self.calls+=1
        if not self.columns:
            return None
        original=kwargs['attention_mask']
        require(original.ndim==4 and original.shape[0:2]==(1,1),'native causal mask unavailable')
        heads=list(range(module.num_heads)) if self.heads is None else self.heads
        mask=original.expand(1,module.num_heads,length,original.shape[-1]).clone()
        floor=self.ev.torch.finfo(mask.dtype).min
        for h in heads:
            require(0<=h<module.num_heads,'head outside architecture')
            for row in self.rows:
                require(bool((mask[0,h,row,self.columns]>=0).all()),'cut touches already masked attention')
                mask[0,h,row,self.columns]=floor
        return args,{**kwargs,'attention_mask':mask}

    def __exit__(self,*args):
        self.handle.remove()


def capture(evaluator,inputs,layers,positions,attention=False):
    states,stats,handles={},{},[]
    n=inputs['input_ids'].shape[1]
    def hidden_hook(index):
        def hook(module,args,out):
            require(out[0].shape[1]==n,'state extraction receives appended query answer')
            states[str(index)]={str(p):out[0][0,p].float().cpu().tolist() for p in positions}
        return hook
    try:
        for layer in layers:
            handles.append(evaluator.model.model.layers[layer].register_forward_hook(hidden_hook(layer)))
        evaluator.synchronize()
        with evaluator.torch.inference_mode():
            out=evaluator.model(**inputs,use_cache=False,return_dict=True,output_attentions=attention)
        if attention:
            # Raw attention is reduced immediately by the caller; full matrices
            # must not enter persistent output or survive another model forward.
            stats=out.attentions
        del out
        return states,stats
    finally:
        for handle in handles:
            handle.remove()


class DaraColumns:
    """Temporary first-layer native forward with the authors' raw-logit multiplier.

    All operations follow pinned Transformers4.49 Qwen2VLAttention, inserting
    image-column scaling before the causal mask. Eight slots cycle modulo8,
    including the query image, as in the pinned official DARA release.
    """
    def __init__(self,evaluator,metadata,factors):
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb,repeat_kv
        self.ev,self.factors=evaluator,factors
        self.rotary,self.repeat=apply_multimodal_rotary_pos_emb,repeat_kv
        ids=metadata['input_token_ids'];token=evaluator.model.config.image_token_id
        self.ranges=[]
        for i,v in enumerate(ids):
            if v==token:
                if i==0 or ids[i-1]!=token:self.ranges.append([])
                self.ranges[-1].append(i)
        require(len(self.ranges)==len(metadata['image_grid_thw']),'DARA image slots malformed')
        require(tuple(factors.shape)==(evaluator.model.config.num_attention_heads,8),'DARA factor shape')
        self.calls=0

    def __enter__(self):
        from types import MethodType
        self.attention=self.ev.model.model.layers[0].self_attn
        require(type(self.attention).__name__=='Qwen2VLAttention','DARA requires pinned native eager attention')
        self.had_instance_forward='forward' in self.attention.__dict__
        self.previous=self.attention.__dict__.get('forward')
        self.attention.forward=MethodType(self.forward,self.attention)
        return self

    def forward(self,a,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
                output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
        t=self.ev.torch
        batch,length,_=hidden_states.shape
        require(batch==1 and hidden_states.dtype in (t.float32,t.bfloat16),'DARA requires batch1 FP32/BF16')
        q=a.q_proj(hidden_states).view(batch,length,-1,a.head_dim).transpose(1,2)
        k=a.k_proj(hidden_states).view(batch,length,-1,a.head_dim).transpose(1,2)
        v=a.v_proj(hidden_states).view(batch,length,-1,a.head_dim).transpose(1,2)
        cos,sin=position_embeddings
        q,k=self.rotary(q,k,cos,sin,a.rope_scaling['mrope_section'])
        if past_key_value is not None:
            k,v=past_key_value.update(k,v,a.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
        k=self.repeat(k,a.num_key_value_groups);v=self.repeat(v,a.num_key_value_groups)
        logits=t.matmul(q,k.transpose(2,3))/math.sqrt(a.head_dim)
        multiplier=t.ones((batch,a.num_heads,1,k.shape[-2]),dtype=logits.dtype,device=logits.device)
        for slot,columns in enumerate(self.ranges):
            require(max(columns)<k.shape[-2],'DARA image position beyond available cache')
            multiplier[:,:,:,columns]=self.factors[:,slot%8].to(logits.dtype).view(1,a.num_heads,1,1)
        logits=logits*multiplier
        if attention_mask is not None:
            logits=logits+attention_mask[:,:,:,:k.shape[-2]]
        weights=t.nn.functional.softmax(logits,dim=-1,dtype=t.float32).to(q.dtype)
        weights=t.nn.functional.dropout(weights,p=a.attention_dropout,training=a.training)
        output=t.matmul(weights,v)
        require(tuple(output.shape)==(batch,a.num_heads,length,a.head_dim),'DARA attention output shape')
        output=a.o_proj(output.transpose(1,2).contiguous().reshape(batch,length,-1))
        self.calls+=1
        return output,weights if output_attentions else None,past_key_value

    def __exit__(self,*args):
        if self.had_instance_forward:self.attention.forward=self.previous
        else:delattr(self.attention,'forward')
