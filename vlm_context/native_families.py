"""Native family scoring from the accepted experiments.

Candidate scoring, processor templates, strict parsing, and prefix validation are
preserved. Model paths are supplied by the caller; campaign admission is absent.
"""
from __future__ import annotations
import contextlib, copy, hashlib, importlib.metadata, math, pathlib, re, sys, time, types
from .io import read_json as read, sha256 as sha, require as need

def digest(value):
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def verify_files(files, root=None):
    root=pathlib.Path(root or '.')
    for name, expected in files.items():
        path=root/name
        if not path.is_file() or sha(path)!=expected:
            raise ValueError('Pinned model file missing or changed: '+name)

CLASSES={'qwen2_vl':'Qwen2VLForConditionalGeneration','qwen2_5_vl':'Qwen2_5_VLForConditionalGeneration','idefics2':'Idefics2ForConditionalGeneration','internvl':'InternVLForConditionalGeneration'}

def boundary(prefix,full,label,decode,eos_ids):
    """Fail closed on any prompt retokenization, truncated label or ambiguous EOS."""
    if not prefix or full[:len(prefix)]!=prefix: raise ValueError('native answer changes shared prefix')
    tail=full[len(prefix):]
    stop=next((i for i,t in enumerate(tail) if t in eos_ids),None)
    if stop is None or stop==0: raise ValueError('native answer must contain full label and an end-of-turn token')
    answer=decode(tail[:stop])
    if answer.strip()!=label or '\n' in answer.strip(): raise ValueError('native answer span does not decode to complete label')
    if any(t in eos_ids for t in tail[:stop]): raise ValueError('early end token')
    return {'prefix_length':len(prefix),'label_token_ids':tail[:stop],
            'scored_token_ids':tail[:stop+1],'native_end_token':tail[stop],
            'decoded_label_span':answer,'trailing_template_token_ids':tail[stop+1:]}

def strict_parse(text,labels):
    text=text.strip()
    return labels.index(text) if text in labels else None

def historical_parse(text,labels):
    hits=[(text.lower().find(label.lower()),i) for i,label in enumerate(labels) if label.lower() in text.lower()]
    return min(hits)[1] if hits else None

def runtime_identity(spec):
    snapshot=pathlib.Path(spec['snapshot'])
    verify_files(spec['snapshot_sha256'],snapshot)
    cfg=read(snapshot/'config.json')
    if cfg.get('model_type')!=spec['model_type']: raise ValueError('model family mismatch; fallback forbidden')
    return {'model_id':spec['model_id'],'revision':spec['revision'],'model_type':spec['model_type'],
            'snapshot_sha256':spec['snapshot_sha256'],'python':sys.version,
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','Pillow','tokenizers','safetensors','peft','accelerate')},
            'dtype':spec['dtype'],'attention':'eager','trust_remote_code':False,'model_fallback':False}

class HFNative:
    def __init__(self,spec,device):
        import torch, transformers
        self.torch=torch;self.spec=spec;self.device=device
        self.identity=runtime_identity(spec);self.identity['device']=device
        if self.identity['packages']['transformers']!=spec['transformers']: raise ValueError('Transformers version differs from recorded runtime')
        cls=getattr(transformers,CLASSES[spec['model_type']])
        self.processor=transformers.AutoProcessor.from_pretrained(spec['snapshot'],local_files_only=True,trust_remote_code=False,use_fast=False)
        self.tokenizer=self.processor.tokenizer
        dtype=getattr(torch,spec['dtype'])
        self.model=cls.from_pretrained(spec['snapshot'],local_files_only=True,trust_remote_code=False,torch_dtype=dtype,attn_implementation='eager').to(device).eval()
        self.model.requires_grad_(False)
        eos=self.model.generation_config.eos_token_id
        self.eos_ids=set(eos if isinstance(eos,list) else [eos])
        self.eos_ids.discard(None)
        if not self.eos_ids: raise ValueError('native generation EOS missing')
        self.identity['processor_class']=type(self.processor).__name__
        self.identity['model_class']=type(self.model).__name__
        self.identity['native_eos_ids']=sorted(self.eos_ids)
        self.identity['chat_template']=getattr(self.processor,'chat_template',None) or self.tokenizer.chat_template
        cfg=self.model.config
        text=getattr(cfg,'text_config',cfg)
        self.context_limit=getattr(text,'max_position_embeddings',None)
        if not isinstance(self.context_limit,int): raise ValueError('unresolved native context limit')
    def decode(self,ids): return self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    def text(self,messages,answer=None):
        msg=copy.deepcopy(messages)
        if answer is not None: msg.append({'role':'assistant','content':[{'type':'text','text':answer}]})
        return self.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=answer is None)
    def encode(self,text,images):
        kw={'text':[text],'return_tensors':'pt','padding':False,'truncation':False}
        if images: kw['images']=images
        out=self.processor(**kw)
        if out['input_ids'].shape[-1]>self.context_limit: raise ValueError('native context limit exceeded; no truncation permitted')
        return {k:v.to(self.device,dtype=self.model.dtype) if v.is_floating_point() else v.to(self.device) for k,v in out.items()}
    def candidates(self,messages,images,labels):
        prompt=self.text(messages);base=self.encode(prompt,images);prefix=base['input_ids'][0].tolist()
        result=[]
        for label in labels:
            complete=self.encode(self.text(messages,label),images)
            b=boundary(prefix,complete['input_ids'][0].tolist(),label,self.decode,self.eos_ids)
            # Full template suffix may have a newline after EOS. Keep it: causal
            # logits before EOS cannot depend on later tokens; smoke checks this.
            result.append((label,complete,b))
        return prompt,base,result
    def score(self,messages,images,labels,validate=False):
        torch=self.torch;start=time.monotonic()
        prompt,base,candidates=self.candidates(messages,images,labels)
        scores=[];checks=[]
        with torch.inference_mode():
            for label,inputs,b in candidates:
                logits=self.model(**inputs,use_cache=False).logits.float()
                p=b['prefix_length'];target=b['scored_token_ids']
                if logits.shape[1]!=inputs['input_ids'].shape[1]: raise ValueError('model logits do not align with native expanded input IDs')
                positions=torch.arange(p-1,p+len(target)-1,device=logits.device)
                ts=torch.tensor(target,device=logits.device)
                logp=logits[0,positions].log_softmax(-1).gather(1,ts[:,None]).flatten()
                if not torch.isfinite(logp).all(): raise ValueError('nonfinite candidate probabilities')
                values=logp.tolist();scores.append({'label':label,**b,'token_log_probabilities':values,'sum_logp':sum(values),'mean_logp':sum(values)/len(values)})
                if validate:
                    # Independent native prefix forwards, reprocessed decoded
                    # strings, avoid a model-family-specific cache shortcut.
                    errors=[];prefix_values=[]
                    for i,token in enumerate(target):
                        prefix_text=prompt+self.decode(b['label_token_ids'][:i])
                        step=self.encode(prefix_text,images)
                        expected=inputs['input_ids'][0,:p+i].tolist()
                        if step['input_ids'][0].tolist()!=expected: raise ValueError('independent prefix retokenizes')
                        v=self.model(**step,use_cache=False).logits[0,-1].float().log_softmax(-1)[token].item()
                        prefix_values.append(v);errors.append(abs(v-values[i]))
                    error=max(errors)
                    if error>0.08: raise ValueError('causal sequence/prefix scoring check failed')
                    checks.append({'label':label,'max_abs_logp_error':error,'token_count':len(target),'prefix_token_log_probabilities':prefix_values})
            generated=self.model.generate(**base,max_new_tokens=16,do_sample=False,use_cache=True)
        ids=generated[0,base['input_ids'].shape[1]:].tolist()
        decoded=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        best=max(range(len(scores)),key=lambda i:scores[i]['sum_logp']); mean=max(range(len(scores)),key=lambda i:scores[i]['mean_logp'])
        return {'prompt':prompt,'prompt_token_ids':base['input_ids'][0].tolist(),
                'native_input_shapes':{k:list(v.shape) for k,v in base.items()},'context_limit':self.context_limit,
                'candidates':scores,'sum_prediction':labels[best],'mean_prediction':labels[mean],
                'greedy_text':decoded,'greedy_token_ids':ids,'strict_prediction_index':strict_parse(decoded,labels),
                'historical_prediction_index':historical_parse(decoded,labels),
                'causal_checks':checks,'elapsed_seconds':time.monotonic()-start}
    def generate(self,messages,images,max_new_tokens=16):
        inputs=self.encode(self.text(messages),images)
        with self.torch.inference_mode(): out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
        ids=out[0,inputs['input_ids'].shape[1]:].tolist()
        return {'greedy_text':self.tokenizer.decode(ids,skip_special_tokens=True),'greedy_token_ids':ids,
                'prompt':self.text(messages),'prompt_token_ids':inputs['input_ids'][0].tolist(),
                'native_input_shapes':{k:list(v.shape) for k,v in inputs.items()}}


def prefix_inputs(engine, complete, length):
    torch = engine.torch
    need(engine.spec['model_type'] == 'idefics2', 'exact-prefix recovery restricted to IDEFICS2')
    need(set(complete) == {'input_ids', 'attention_mask', 'pixel_values', 'pixel_attention_mask'},
         'unexpected IDEFICS2 processor keys')
    ids, mask = complete['input_ids'], complete['attention_mask']
    need(ids.ndim == mask.ndim == 2 and ids.shape == mask.shape and ids.shape[0] == 1,
         'unexpected native text dimensions')
    need(0 < length <= ids.shape[1] and bool(torch.all(mask == 1)),
         'prefix must be an unpadded native text prefix')
    return {key: value[:, :length] if key in ('input_ids', 'attention_mask') else value
            for key, value in complete.items()}

def install_exact_prefix(engine):
    need(engine.spec['model_type'] == 'idefics2', 'IDEFICS2 only')
    original = engine.score

    def score(messages, images, labels, validate=False):
        # This is the original complete-label + native EOS score and generation.
        result = original(messages, images, labels, validate=False)
        if not validate:
            return result
        prompt, base, candidates = engine.candidates(messages, images, labels)
        need(prompt == result['prompt'] and base['input_ids'][0].tolist() == result['prompt_token_ids'],
             'independent native processor replay differs')
        need(len(candidates) == len(result['candidates']) == len(labels), 'candidate coverage differs')
        checks = []
        with engine.torch.inference_mode():
            for (label, inputs, boundary), frozen in zip(candidates, result['candidates']):
                need(frozen['label'] == label and all(frozen[key] == value for key, value in boundary.items()),
                     'independent complete-answer boundary differs')
                for key in ('pixel_values', 'pixel_attention_mask'):
                    need(key in base and key in inputs and engine.torch.equal(base[key], inputs[key]),
                         'native candidate changes visual inputs')
                p = boundary['prefix_length']
                target = boundary['scored_token_ids']
                need(len(target) == len(frozen['token_log_probabilities']) and
                     inputs['input_ids'][0, p:p+len(target)].tolist() == target,
                     'complete-label plus EOS token identity differs')
                values = []
                for index, token in enumerate(target):
                    step = prefix_inputs(engine, inputs, p + index)
                    need(step['input_ids'][0].tolist() == inputs['input_ids'][0, :p+index].tolist(),
                         'native exact prefix identity differs')
                    value = engine.model(**step, use_cache=False).logits[0, -1].float().log_softmax(-1)[token].item()
                    need(math.isfinite(value), 'nonfinite independent causal probability')
                    values.append(value)
                error = max(abs(a-b) for a,b in zip(values, frozen['token_log_probabilities']))
                need(error <= .08, 'causal sequence/prefix scoring check failed')
                checks.append({'label': label, 'max_abs_logp_error': error, 'token_count': len(target),
                               'prefix_token_log_probabilities': values,
                               'prefix_construction': 'exact native token prefix; unchanged full visual inputs'})
        result['causal_checks'] = checks
        return result
    engine.score = score
    return engine

@contextlib.contextmanager
def official_clip_loader(profile,modules=None):
 if modules is None:
  import open_clip,open_clip.openai
  need(importlib.metadata.version('open_clip_torch')=='2.20.0','OpenCLIP version differs from recorded runtime')
  modules=(open_clip,open_clip.openai)
 clip,openai=modules;checkpoint=pathlib.Path(profile['checkpoint'])
 need(checkpoint.is_file() and sha(checkpoint)==profile['checkpoint_sha256'],'pinned OpenAI archive changed')
 original_create=clip.create_model_and_transforms;original_download=openai.download_pretrained_from_url
 def offline_download(url,cache_dir=None):
  need(url==profile['official_url'],'unexpected OpenAI weight URL')
  need(pathlib.Path(cache_dir).resolve()==checkpoint.parent.resolve(),'unexpected OpenAI cache directory')
  need(sha(checkpoint)==profile['checkpoint_sha256'],'OpenAI archive changed at loading boundary')
  return str(checkpoint)
 def native_create(model_name,pretrained=None,cache_dir=None,**kwargs):
  need(model_name=='ViT-L-14' and pretrained==str(checkpoint),'unregistered Flamingo vision request')
  need(cache_dir is None and not kwargs,'unexpected native constructor override')
  return original_create(model_name,pretrained='openai',cache_dir=str(checkpoint.parent))
 clip.create_model_and_transforms=native_create;openai.download_pretrained_from_url=offline_download
 try:yield
 finally:
  clip.create_model_and_transforms=original_create;openai.download_pretrained_from_url=original_download

def same_view(left, right):
    return (left.device == right.device and left.numel() > 0 and right.numel() > 0
            and left.data_ptr() == right.data_ptr() and left.shape == right.shape
            and left.dtype == right.dtype and left.stride() == right.stride())

def augment(module, state, torch):
    current = module.state_dict()
    required = {name for name in current if 'gated_cross_attn' in name or 'perceiver' in name}
    need(required, 'native multimodal inventory absent')
    need(not (set(state) - set(current)), 'unexpected checkpoint tensor key')
    provided = set(state) & set(current)
    augmented = copy.copy(state)
    aliases = {}
    for target in sorted(required - provided):
        sources = sorted(name for name in provided if same_view(current[target], current[name]))
        need(sources, 'genuine multimodal checkpoint tensor missing: ' + target)
        source = sources[0]
        value = state[source]
        need(value.shape == current[target].shape and value.dtype == current[target].dtype,
             'alias checkpoint tensor shape/dtype differs')
        need(all(torch.equal(value, state[name]) for name in sources), 'conflicting provided tensor aliases')
        augmented[target] = value
        need(augmented[target] is state[source] and torch.equal(augmented[target], state[source]),
             'alias fill changed original checkpoint values')
        aliases[target] = {'provided_alias': source, 'identical_storage': True,
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'original_checkpoint_tensor_reused': True, 'checkpoint_values_equal_after_load': False}
    return augmented, aliases, required

class FlamingoNative:
    def __init__(self,spec,device):
        import torch
        self.torch=torch;self.spec=spec;self.device=device;rt=spec['family_runtime']
        verify_files(spec['snapshot_sha256'],spec['snapshot'])
        sys.path.insert(0,rt['code_path'])
        from open_flamingo import create_model_and_transforms
        if rt['language_model_id']!='togethercomputer/RedPajama-INCITE-Base-3B-v1' or rt['vision_encoder']!='ViT-L-14' or rt['cross_attn_every_n_layers']!=2:raise ValueError('incorrect OpenFlamingo4B backbone contract')
        with official_clip_loader(rt['clip_profile']):
            self.model,self.image_processor,self.tokenizer=create_model_and_transforms(clip_vision_encoder_path='ViT-L-14',clip_vision_encoder_pretrained=rt['vision_checkpoint'],lang_encoder_path=rt['language_snapshot'],tokenizer_path=rt['language_snapshot'],cross_attn_every_n_layers=2,use_local_files=True)
        weights=pathlib.Path(spec['snapshot'])/'checkpoint.pt'
        state=torch.load(weights,map_location='cpu',weights_only=True)
        state,aliases,required=augment(self.model,state,torch)
        if 'expected_aliases' in rt and set(aliases)!=set(rt['expected_aliases']): raise ValueError('Pinned multimodal alias inventory changed')
        incompat=self.model.load_state_dict(state,strict=False)
        required={k for k in self.model.state_dict() if 'gated_cross_attn' in k or 'perceiver' in k}
        if incompat.unexpected_keys or required.intersection(incompat.missing_keys):raise ValueError('OpenFlamingo multimodal weights missing/unexpected')
        self.model=self.model.to(device=device,dtype=getattr(torch,spec['dtype'])).eval();self.model.requires_grad_(False)
        self.eos_ids={self.tokenizer.encode('<|endofchunk|>',add_special_tokens=False)[-1]}
        self.context_limit=self.model.lang_encoder.config.max_position_embeddings
        self.identity={'model_id':spec['model_id'],'revision':spec['revision'],'runtime':rt,'family':'open_flamingo','missing_frozen_backbone_keys':incompat.missing_keys,'device':device,'native_eos_ids':sorted(self.eos_ids),'model_fallback':False,'native_end_token':'<|endofchunk|>','dtype':spec['dtype']}
    def prompt(self,messages):
        # Native Flamingo interleaving uses end-of-chunk between support images,
        # not a Qwen/ChatML conversation or universal EOS convention.
        content=messages[0]['content'];parts=[]
        for i,item in enumerate(content):
            if item['type']=='image':parts.append('<image>')
            else:
                parts.append(item['text'])
                if i<len(content)-1:parts.append('<|endofchunk|>')
        return ''.join(parts)+' Answer:'
    def inputs(self,prompt,images):
        t=self.tokenizer(prompt,return_tensors='pt',add_special_tokens=True)
        if t['input_ids'].shape[-1]>self.context_limit:raise ValueError('OpenFlamingo context limit exceeded')
        vision=self.torch.stack([self.image_processor(i) for i in images])[None,:,None].to(self.device,dtype=getattr(self.torch,self.spec['dtype']))
        return {'vision_x':vision,'lang_x':t['input_ids'].to(self.device),'attention_mask':t['attention_mask'].to(self.device)}
    def score(self,messages,images,labels,validate=False):
        torch=self.torch;start=time.monotonic();prompt=self.prompt(messages);base=self.inputs(prompt,images);prefix=base['lang_x'][0].tolist();rows=[];checks=[]
        with torch.inference_mode():
            for label in labels:
                # Literal space is part of this native completion convention.
                complete=self.inputs(prompt+' '+label+'<|endofchunk|>',images)
                span=boundary(prefix,complete['lang_x'][0].tolist(),label,lambda ids:self.tokenizer.decode(ids,skip_special_tokens=False),self.eos_ids)
                p=len(prefix);ts=span['scored_token_ids'];logits=self.model(**complete,clear_conditioned_layers=True,use_cache=False).logits.float()
                values=[logits[0,p+i-1].log_softmax(-1)[t].item() for i,t in enumerate(ts)]
                rows.append({'label':label,**span,'token_log_probabilities':values,'sum_logp':sum(values),'mean_logp':sum(values)/len(values)})
                if validate:
                    errors=[];prefix_values=[]
                    for i,t in enumerate(ts):
                        step={**complete,'lang_x':complete['lang_x'][:,:p+i],'attention_mask':complete['attention_mask'][:,:p+i]}
                        v=self.model(**step,clear_conditioned_layers=True,use_cache=False).logits[0,-1].float().log_softmax(-1)[t].item();prefix_values.append(v);errors.append(abs(v-values[i]))
                    if max(errors)>.08:raise ValueError('Flamingo causal score disagreement')
                    checks.append({'label':label,'max_abs_logp_error':max(errors),'token_count':len(ts),'prefix_token_log_probabilities':prefix_values})
            out=self.model.generate(**base,max_new_tokens=16,num_beams=1,do_sample=False)
        ids=out[0,len(prefix):].tolist();text=self.tokenizer.decode(ids,skip_special_tokens=True)
        return {'prompt':prompt,'prompt_token_ids':prefix,'native_input_shapes':{k:list(v.shape) for k,v in base.items()},'context_limit':self.context_limit,'candidates':rows,'sum_prediction':max(rows,key=lambda r:r['sum_logp'])['label'],'mean_prediction':max(rows,key=lambda r:r['mean_logp'])['label'],'greedy_text':text,'greedy_token_ids':ids,'strict_prediction_index':strict_parse(text,labels),'historical_prediction_index':historical_parse(text,labels),'causal_checks':checks,'elapsed_seconds':time.monotonic()-start}

def position_ids_cpu(attention_mask, past_key_values):
    # Same integer algorithm as transformers 4.28.1 GPTNeoX, on CPU only.
    mask = attention_mask.to(device='cpu')
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    if past_key_values:
        positions = positions[:, -1].unsqueeze(-1)
    return positions

def prepare_exact(original, input_ids, past_key_values, attention_mask, kwargs, *, override):
    """The original method still slices input IDs and returns all native fields."""
    if override and attention_mask is not None and kwargs.get('position_ids') is None:
        kwargs = dict(kwargs)
        kwargs['position_ids'] = position_ids_cpu(attention_mask, past_key_values).to(
            device=attention_mask.device)
    return original(input_ids, past_key_values=past_key_values,
                    attention_mask=attention_mask, **kwargs)
def install_generation_positions(engine):
    """Same integer position algorithm as pinned GPTNeoX, needed only on MPS."""
    if engine.device!='mps':return engine
    need(importlib.metadata.version('transformers')=='4.28.1','OpenFlamingo uses Transformers4.28.1')
    encoder=engine.model.lang_encoder; original=encoder.prepare_inputs_for_generation
    def prepare(self,input_ids,past_key_values=None,attention_mask=None,**kwargs):
        return prepare_exact(original,input_ids,past_key_values,attention_mask,kwargs,
            override=attention_mask is not None and attention_mask.device.type=='mps')
    encoder.prepare_inputs_for_generation=types.MethodType(prepare,encoder)
    return engine

def load_native(spec,model_dir,device,*,checkpoint=None,inference_dtype=None,family_runtime=None):
    """Load pinned model bytes using an explicit local model directory."""
    spec=copy.deepcopy(spec);spec['snapshot']=str(pathlib.Path(model_dir).expanduser().resolve())
    if spec['model_type']=='open_flamingo':
        need(family_runtime is not None,'OpenFlamingo requires explicit pinned family_runtime paths; see NATIVE_EXPERIMENTS.md')
        spec['family_runtime']=family_runtime
        engine=install_generation_positions(FlamingoNative(spec,device))
    else:
        engine=HFNative(spec,device)
        if spec['model_type']=='idefics2':install_exact_prefix(engine)
    if inference_dtype:
        engine.model.to(dtype=getattr(engine.torch,inference_dtype))
        engine.identity.update(load_dtype=spec['dtype'],dtype=inference_dtype)
    if checkpoint:
        from peft import PeftModel
        engine.model=PeftModel.from_pretrained(engine.model,str(checkpoint),is_trainable=False,local_files_only=True).get_base_model().eval()
        engine.model.requires_grad_(False)
    return engine

def classification_messages(episode,loader):
    images=[];content=[]
    for support,label in zip(episode['supports'],episode['support_labels']):
        images.append(loader(support['image_id']));content += [{'type':'image'},{'type':'text','text':'This image shows: '+label+'.'}]
    images.append(loader(episode['query']['image_id']))
    content += [{'type':'image'},{'type':'text','text':'Classify the last image. Answer with exactly one of ['+', '.join(episode['displayed_options'])+'] and nothing else.'}]
    return [{'role':'user','content':content}],images
