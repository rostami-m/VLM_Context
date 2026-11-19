"""Accepted FP32 postmerger vision-feature comparison."""
import math
from vlm_context.io import write_json as publish,require,sha256 as sha,read_json,source_path
def schedule(session): return read_json(source_path("data/protocols/interventions/baseline_schedule.json.gz"))


def pool(raw):
    require(len(raw)==16 and all(len(r)==1536 and all(math.isfinite(v) for v in r) for r in raw),'postmerger feature shape/value mismatch')
    mean=[math.fsum(r[j] for r in raw)/16 for j in range(1536)]
    norm=math.sqrt(math.fsum(v*v for v in mean));require(norm>0 and math.isfinite(norm),'zero/nonfinite feature norm')
    return {'mean':mean,'norm':norm,'unit':[v/norm for v in mean]}



def features(session,smoke_only=False):
    session.start_model();ev=session.evaluator;torch=ev.torch
    for parameter in ev.model.parameters():parameter.requires_grad_(False)
    all_schedule=schedule(session);records={}
    for task in ('cifar4','fashion4'):
        manifest,loader=session.loader(task)
        if smoke_only:
            ids=sorted(manifest['roles']['development'])[:2]
        else:
            # The requested non-ICL comparator uses exactly16 demonstrations.
            selected=[p for p in all_schedule['panels'] if p['episodes'][0]['task']==task and p['episodes'][0]['k']==16]
            ids=sorted({e['query']['image_id'] for p in selected for e in p['episodes']}|
                       {s['image_id'] for p in selected for s in p['episodes'][0]['supports']})
        for index,image_id in enumerate(ids):
            im=loader(image_id);direct=ev.processor.image_processor(images=[im],return_tensors='pt').to(ev.device)
            require(list(im.size)==[112,112] and direct['image_grid_thw'].tolist()==[[1,8,8]],'fixed feature image geometry changed')
            with torch.inference_mode():
                raw=ev.model.visual(direct['pixel_values'].type(ev.model.visual.get_dtype()),grid_thw=direct['image_grid_thw'])
            require(str(raw.dtype)=='torch.float32' and list(raw.shape)==[16,1536],'actual postmerger feature dtype/shape')
            values=raw.float().cpu().tolist();proof=None
            if smoke_only:
                template=next(f['episode'] for f in all_schedule['smoke_fixtures'] if f['episode']['task']==task and f['episode']['k']==0)
                import copy
                ep=copy.deepcopy(template);ep['query']={k:manifest['images'][image_id][k] for k in ('image_id','class_id','content_sha256')}
                from vlm_context.qwen import render_messages
                messages,images=render_messages(ep,loader);inputs,metadata=ev.encode(messages,images,ep['label_space'])
                require(all(torch.equal(direct[k],inputs[k]) for k in ('pixel_values','image_grid_thw')),'feature image/Q0 processor mismatch')
                captures=[]
                handle=ev.model.visual.register_forward_hook(lambda m,a,o:captures.append(o.float().cpu().tolist()))
                try:
                    with torch.inference_mode():out=ev.model(**inputs,use_cache=False,return_dict=True)
                    require(bool(torch.isfinite(out.logits).all()),'nonfinite feature native model logits')
                    del out
                finally:handle.remove()
                require(captures==[values],'direct postmerger tokens differ from native Q0 hook')
                proof={'direct_Q0_pixels_grid_exact':True,'hook_tokens':captures[0],'Q0_episode':ep,'Q0_encoding':metadata}
            record={'image_id':image_id,'image':manifest['images'][image_id],'task':task,'token_features':values,
                    'dtype':str(raw.dtype),'shape':list(raw.shape),'pool':pool(values),'native_equivalence':proof,
                    'feature_definition':'postmerger16tokens from FP32 evaluation of the exact BF16 parent snapshot; mean in float64 then L2 norm; fixed112 RGB'}
            session.save(f'feature_{task}_{index:03d}',record);records[image_id]=record
    if not smoke_only:
        from vlm_context.kernels.gate.policy import cosine_nearest
        summaries=[]
        for panel in all_schedule['panels']:
            if panel['episodes'][0]['k']!=16:continue
            rows=[]
            for ep in panel['episodes']:
                neighbors=[{'image_id':s['image_id'],'class_id':s['class_id'],'unit':records[s['image_id']]['pool']['unit']} for s in ep['supports']]
                nearest=cosine_nearest(records[ep['query']['image_id']]['pool']['unit'],neighbors)
                rows.append({'query_id':ep['query']['image_id'],'nearest':nearest,
                             'predicted_label':ep['label_space'][nearest['class_id']],
                             'correct':nearest['class_id']==ep['query']['class_id']})
            summaries.append({'panel_id':panel['panel_id'],'accuracy':sum(r['correct'] for r in rows)/80,'rows':rows})
        publish(session.output/'summary.json',{'kind':'q08-own-vision-1nn-v1','panels':summaries,
                'feature_count':len(records),'training':False,'K0':'not applicable without demonstrations',
                'label_use':'episode support labels only; no target-label fitting',
                'evaluation_precision':'float32'})
    return session.finish({'feature_count':len(records),'development_only':smoke_only,
                          'summary_sha256':None if smoke_only else sha(session.output/'summary.json')})
