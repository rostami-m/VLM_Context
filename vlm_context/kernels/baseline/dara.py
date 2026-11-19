"""Source-derived DARA factors, raw-logit intervention and fixed training recipe."""
import math,time
from vlm_context.kernels.gate.common import digest,score
from vlm_context.io import read_json as read,write_json as publish,require,sha256 as sha,source_path
from vlm_context.kernels.inject.hooks import DaraColumns,classification_renderer,encode


RECIPE={'steps':150,'technical_steps':2,'batch_size':1,'gradient_accumulation':1,
        'optimizer':'AdamW FP32','learning_rate':.001,'gradient_scale':.1,'weight_decay':0.,
        'betas':[.9,.999],'epsilon':1e-8,'warmup_steps':5,'schedule':'linear_after_warmup',
        'factor_slots':8,'layer':0,'initial_factor':1.,'all_heads':True,
        'checkpoint_selection':'final fixed step only; no outcome or minimum-training-loss selection',
        'adaptations':'Qwen2-VL2B, role-disjoint Shapes,150 same-source updates, full-label+EOS loss, native AdamW, batch1/accumulation1',
        'official_source_revision':'36e8a1910b3eb2aa6e339ca19cca32bbb27e8edc'}



def factor_tensor(session,values=None,trainable=False):
    torch=session.evaluator.torch
    shape=(session.evaluator.model.config.num_attention_heads,8)
    t=torch.ones(shape,device=session.evaluator.device,dtype=torch.float32) if values is None else torch.tensor(values,device=session.evaluator.device,dtype=torch.float32)
    require(tuple(t.shape)==shape and bool(torch.isfinite(t).all()),'invalid DARA factor bank')
    return torch.nn.Parameter(t,requires_grad=True) if trainable else t



def scored(session,ep,loader,factors,native=False):
    session.evaluator.synchronize();started=time.monotonic()
    inputs,metadata=encode(session.evaluator,ep,loader)
    with DaraColumns(session.evaluator,metadata,factors) as hook:
        result=score(session,ep,loader,native=native,renderer=classification_renderer,prepared=(inputs,metadata))
    session.evaluator.synchronize();result['seconds']=time.monotonic()-started
    require(hook.calls>0,'DARA did not execute')
    result['dara']={'factors':factors.detach().float().cpu().tolist(),'layer':0,
                    'slot_rule':'image order modulo8 including query','attention_calls':hook.calls}
    return result



def train_steps(session,smoke=False):
    from vlm_context.adaptation import training_batch
    ev=session.evaluator;torch=ev.torch
    manifest,loader=session.loader('shapes4');objective=session.args.condition or 'episode_remap'
    require(objective in ('natural','episode_remap'),'DARA objective not declared')
    episodes=read(source_path('data/protocols/q05/training_schedule.json.gz'))['shapes4'][str(session.args.seed or 0)]['training'][objective]
    require(len(episodes)==150,'training scope differs from Q05 source budget')
    steps=2 if smoke else 150
    flags=[(p,p.requires_grad) for p in ev.model.parameters()]
    for p,_ in flags:p.requires_grad_(False)
    factors=factor_tensor(session,trainable=True);factors.register_hook(lambda g:g*RECIPE['gradient_scale'])
    optimizer=torch.optim.AdamW([factors],lr=RECIPE['learning_rate'],betas=tuple(RECIPE['betas']),
                                eps=RECIPE['epsilon'],weight_decay=RECIPE['weight_decay'])
    started=time.monotonic();records=[];ev.model.eval()
    try:
        for index,ep in enumerate(episodes[:steps]):
            rate=RECIPE['learning_rate']*min((index+1)/5.,(150-index)/145.)
            optimizer.param_groups[0]['lr']=rate;optimizer.zero_grad(set_to_none=True)
            batch,metadata=training_batch(ev,ep,loader)
            previous=factors.detach().clone()
            with DaraColumns(ev,metadata['encoding'],factors):
                out=ev.model(**batch,use_cache=False,return_dict=True)
            loss=out.loss
            require(bool(torch.isfinite(loss)),'nonfinite DARA training loss')
            # Verify the complete answer+EOS causal-shift supervision independently.
            positions=[i for i,v in enumerate(metadata['supervision']['labels']) if v!=-100]
            target_logps=[float(out.logits[0,p-1].float().log_softmax(-1)[metadata['supervision']['labels'][p]].detach().item()) for p in positions]
            require(abs(float(loss.detach())+math.fsum(target_logps)/len(target_logps))<=.05,'DARA loss does not cover full answer')
            loss.backward()
            require(factors.grad is not None and bool(torch.isfinite(factors.grad).all()),'nonfinite/missing DARA gradients')
            gradient=float(factors.grad.float().norm().item());require(gradient>0,'zero factor gradient')
            optimizer.step()
            require(bool(torch.isfinite(factors).all()),'nonfinite DARA factors')
            update=float((factors.detach()-previous).norm().item());require(update>0,'DARA factor update vanished')
            require(all(p.grad is None for p,_ in flags),'frozen base received training gradients')
            record={'step':index,'objective':objective,'seed':session.args.seed or 0,'learning_rate':rate,
                    'loss':float(loss.detach()),'gradient_l2':gradient,'update_l2':update,
                    'target_logps':target_logps,'target_positions':positions,'training':metadata,
                    'factors':factors.detach().cpu().tolist(),'frozen_base_gradients_absent':True}
            session.save(f'dara_train_{index:03d}',record);records.append(record)
            del out,loss
        values=factors.detach().cpu().tolist()
        checkpoint={'kind':'dara-factor-checkpoint-v1','bundle_sha256':session.bundle_sha,
                    'runtime_sha256':session.runtime_sha,'objective':objective,'seed':session.args.seed or 0,
                    'completed_steps':steps,'technical_only':smoke,'factors':values,
                    'trainable_parameters':factors.numel(),'recipe':RECIPE,
                    'source_training_episode_sha256':digest(episodes),'seconds':time.monotonic()-started}
        publish(session.output/'checkpoint.json',checkpoint)
        # Fresh tensor reload, unmodified model and same exact development query.
        # Separate process reload is also a required registered job for full fits.
        reloaded=factor_tensor(session,read(session.output/'checkpoint.json')['factors'])
        require(torch.equal(reloaded,factors.detach()),'DARA checkpoint tensor roundtrip mismatch')
        return {'checkpoint_sha256':sha(session.output/'checkpoint.json'),'updates':steps,'fresh_tensor_equal':True}
    finally:
        for p,flag in flags:p.requires_grad_(flag)
