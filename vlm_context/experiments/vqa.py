"""Q11 open-answer and multiple-choice transfer, without target-task fitting."""
from pathlib import Path
from collections import defaultdict
from ..io import read_json,write_json,source_path,require
from ..native_families import load_native
from ..native_checkpoints import validate_q11
from ..open_tasks import vqa_score,install_bounded_attention,open_image_loader,prepare_data


def scheduled(config,args):
    task=getattr(args,'task',None) or config.get('task','vqav2')
    require(task in ('vqav2','textvqa','scienceqa','rule'),'Unknown Q11 task')
    retained=task in ('textvqa','scienceqa')
    name='retention_episodes.json' if retained else 'episodes.json.gz'
    rows=read_json(source_path('data/protocols/q11/'+name))
    partition='development' if args.smoke else 'confirmatory'
    rows=[e for e in rows if e['task']==task and e['partition']==partition and (retained or e['seed']==args.seed)]
    require(rows,'Selected Q11 schedule is empty')
    if args.smoke:
        # Retain both rule variants instead of breaking a paired metric.
        rows=[e for e in rows if e.get('pair_id')==rows[0]['pair_id']] if task=='rule' else rows[:2]
    return task,rows


def summarize(records,task):
    groups=defaultdict(list)
    if task=='rule':
        pairs=defaultdict(dict)
        for row in records:
            e=row['episode'];key=(e['explicit'],e['nonce'],e['k'],e['pair_id'])
            require(e['rule'] not in pairs[key],'Duplicate counterfactual rule')
            pairs[key][e['rule']]=row['score']
        for (explicit,nonce,k,pair),scores in pairs.items():
            require(set(scores)=={'left','right'},'Rule score requires both counterfactual answers')
            groups[(explicit,nonce,k)].append(float(all(scores.values())))
        return {'cells':[{'explicit':a,'nonce':b,'k':k,'pairs':len(v),'paired_both_rules_accuracy':sum(v)/len(v)} for(a,b,k),v in sorted(groups.items())]}
    for row in records:groups[row['episode']['k']].append(row['score'])
    return {'cells':[{'k':k,'queries':len(v),'score':sum(v)/len(v)} for k,v in sorted(groups.items())]}


def run(config,args):
    task,rows=scheduled(config,args)
    if getattr(args,'mode','evaluate')=='prepare':return prepare_data(args.data_root,task)
    require(getattr(args,'mode','evaluate')!='train','Q11 reuses Shapes adapters and has no task-specific fitting')
    out=Path(args.output);require(not out.exists() or not any(out.iterdir()),'Use a fresh output directory')
    arm=getattr(args,'arm',None) or config.get('arm','frozen')
    spec=read_json(source_path('data/protocols/q11/spec.json'));spec['dtype']='bfloat16'
    validate_q11(args.checkpoint,arm=arm,seed=args.seed,spec=spec,smoke=args.smoke)
    engine=load_native(spec,args.model_dir,args.device,checkpoint=args.checkpoint,inference_dtype='float32')
    engine.identity['bounded_visual_attention']=install_bounded_attention(engine.model,256)
    write_json(out/'runtime.json',engine.identity)
    loader=open_image_loader(args.data_root,task);records=[]
    for n,e in enumerate(rows):
        images=[loader(i) for i in e['images']]
        result=engine.score(e['messages'],images,e['labels'],validate=args.smoke) if 'labels'in e else engine.generate(e['messages'],images,e['max_new_tokens'])
        score=float(result['sum_prediction']==e['answers'][0]) if 'labels'in e else vqa_score(result['greedy_text'],e['answers'])
        row={'episode':e,'result':result,'score':score,'fit_seed':args.seed,'arm':arm,'smoke_only':args.smoke}
        write_json(out/'records'/f'{n:06d}.json',row);records.append(row)
        if n%20==0:print(f'evaluated {n+1}/{len(rows)}',flush=True)
    result=summarize(records,task);write_json(out/'summary.json',result)
    write_json(out/'run.json',{'study':'q11','task':task,'arm':arm,'seed':args.seed,'n':len(rows),'smoke_only':args.smoke,'base_load_dtype':'bfloat16','inference_dtype':'float32'})
    return result
