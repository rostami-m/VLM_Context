"""Q13 recipe: 75natural+75remap, rank8, alpha16, zero dropout/decay."""
from pathlib import Path
import random
from .io import write_json,require
from .native_families import classification_messages
PROJECTIONS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']

def targets(model,family,arm):
    import torch
    modules=dict(model.named_modules())
    if family=='qwen2_5_vl':lm='model.layers';connector=['visual.merger.mlp.0','visual.merger.mlp.2']
    elif family=='internvl':lm='model.language_model.layers';connector=['model.multi_modal_projector.linear_1','model.multi_modal_projector.linear_2']
    else:raise ValueError('Q13 restricted to registered model families')
    n=len(modules[lm]);start={'early':0,'middle':(n-4)//2,'late':n-4}.get(arm)
    names=connector if arm=='connector' else [f'{lm}.{i}.{group}.{proj}' for i in range(start,start+4) for group,ps in [('self_attn',PROJECTIONS[:4]),('mlp',PROJECTIONS[4:])] for proj in ps]
    if len(set(names))!=len(names) or any(name not in modules or not isinstance(modules[name],torch.nn.Linear) for name in names):raise ValueError('native connector/depth topology mismatch')
    return names,{name:[modules[name].in_features,modules[name].out_features] for name in names}

def fit(engine,episodes,loader,arm,output,seed,smoke=False):
    import torch
    from peft import LoraConfig,get_peft_model
    require(arm in ('connector','late'),'Reported Q13 arms are connector and late')
    rows=[r for r in episodes if r['fit_seed']==seed]
    require(len(rows)==150 and sum(r['condition']=='standard' for r in rows)==75 and sum(r['condition']=='remap' for r in rows)==75,'Q13 frozen mixed schedule differs')
    if smoke:rows=rows[:2]
    # RNG is seeded before native model construction, exactly as in the accepted fitter.
    names,shapes=targets(engine.model,engine.spec['model_type'],arm)
    engine.model=get_peft_model(engine.model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.0,target_modules=names,bias='none'))
    engine.model.train();trainable={n:p for n,p in engine.model.get_base_model().named_parameters() if p.requires_grad}
    for parameter in trainable.values():parameter.data=parameter.data.float()
    require(trainable and all('lora_'in name for name in trainable),'Unexpected trainable parameters')
    count=sum(p.numel()for p in trainable.values());require(count==8*sum(sum(s)for s in shapes.values()),'Q13 target capacity differs')
    from .training_primitives import parameter_hashes
    base_hashes={n:h for n,h in parameter_hashes(engine.model.get_base_model(),torch).items() if 'lora_' not in n}
    optimizer=torch.optim.AdamW(trainable.values(),lr=1e-4,weight_decay=0.0)
    output=Path(output);require(not (output/'checkpoint').exists(),'Choose a fresh training output')
    write_json(output/'recipe.json',{'arm':arm,'seed':seed,'rank':8,'alpha':16,'dropout':0,'weight_decay':0,'learning_rate':1e-4,'gradient_clip':1,'steps':len(rows),'trainable_parameters':count,'target_shapes':shapes,'smoke':smoke})
    for index,e in enumerate(rows):
        messages,images=classification_messages(e,loader);gold=e['label_space'][e['query']['class_id']]
        _,_,candidates=engine.candidates(messages,images,[gold]);_,inputs,span=candidates[0]
        labels=torch.full_like(inputs['input_ids'],-100);p=span['prefix_length'];n=len(span['scored_token_ids']);labels[:,p:p+n]=inputs['input_ids'][:,p:p+n]
        optimizer.zero_grad(set_to_none=True);forward=engine.model(**inputs,labels=labels,use_cache=False);loss=forward.loss
        manual=torch.nn.functional.cross_entropy(forward.logits[0,p-1:p+n-1].float(),labels[0,p:p+n])
        require(abs(float(loss.detach())-float(manual.detach()))<=.005,'Complete-answer causal loss mismatch');require(bool(torch.isfinite(loss)),'Nonfinite loss')
        loss.backward();require(all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in trainable.values()),'Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(trainable.values(),1.0);optimizer.step()
        write_json(output/'steps'/f'{index:04d}.json',{'episode_id':e['id'],'loss':float(loss.detach()),'manual_loss':float(manual.detach()),'answer_span':span})
    saved=parameter_hashes(engine.model.get_base_model(),torch)
    require({n:h for n,h in saved.items() if 'lora_' not in n}==base_hashes,'Frozen base parameters changed')
    require(all(bool(torch.isfinite(p).all()) for p in trainable.values()),'Nonfinite adapter')
    write_json(output/'base_unchanged.json',{'before':base_hashes,'after':{n:h for n,h in saved.items() if 'lora_' not in n}})
    engine.model.eval();engine.model.save_pretrained(output/'checkpoint',safe_serialization=True)
    from .io import read_json,source_path
    dev=[e for e in read_json(source_path('data/protocols/replication/smoke.json')) if e['task']=='shapes']
    prior=[]
    for e in dev:
        messages,images=classification_messages(e,loader)
        prior.append(engine.score(messages,images,e['displayed_options']))
    # Release the old model before the exact tensor and prediction reload check.
    import gc
    del optimizer,trainable,forward,loss,manual,engine.model
    gc.collect()
    if engine.device=='mps':torch.mps.empty_cache()
    if engine.device.startswith('cuda'):torch.cuda.empty_cache()
    from .native_families import load_native
    restored=load_native(engine.spec,engine.spec['snapshot'],engine.device,checkpoint=output/'checkpoint')
    require(parameter_hashes(restored.model,torch)==saved,'Fresh reload changed parameter bytes')
    checks=[]
    for e,before in zip(dev,prior):
        messages,images=classification_messages(e,loader);after=restored.score(messages,images,e['displayed_options'])
        error=max(abs(a['sum_logp']-b['sum_logp']) for a,b in zip(before['candidates'],after['candidates']))
        require(error<=.08 and before['sum_prediction']==after['sum_prediction'],'Fresh checkpoint prediction drift')
        checks.append({'episode_id':e['id'],'max_abs_sum_logp_error':error})
    engine.model=restored.model
    write_json(output/'reload.json',{'exact_parameter_match':True,'prediction_checks':checks})
    return output/'checkpoint'
