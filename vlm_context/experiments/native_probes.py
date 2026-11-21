"""Frozen native-family panels, with the accepted model-specific scoring."""
from pathlib import Path
from ..io import read_json,write_json,source_path,require
from ..data import image_loader
from ..native_families import load_native,classification_messages,verify_files


def runtime_paths(config,args):
    if 'family_runtime' not in config:return None
    rt=read_json(source_path(config['family_runtime']))
    root=Path(args.data_root)/'openflamingo'
    rt['code_path']=str(root/'open_flamingo_source')
    rt['language_snapshot']=str(root/'language')
    rt['vision_checkpoint']=str(root/'ViT-L-14.pt')
    rt['clip_profile']['checkpoint']=rt['vision_checkpoint']
    verify_files(rt['language_sha256'],rt['language_snapshot'])
    verify_files(rt['code_sha256'],rt['code_path'])
    return rt


def scheduled(config,args):
    task=getattr(args,'task',None) or config.get('task','cifar4')
    seed=getattr(args,'seed',0)
    rows=read_json(source_path(config['episodes']))
    rows=[e for e in rows if e['task']==task and e['seed']==seed]
    require(rows,'No accepted native episodes for task/seed')
    return task,rows[:2] if args.smoke else rows


def evaluate(engine,rows,loader,out,*,smoke=False):
    groups={}
    for n,e in enumerate(rows):
        messages,images=classification_messages(e,loader)
        result=engine.score(messages,images,e['displayed_options'],validate=n==0)
        gold=e['label_space'][e['query']['class_id']]
        correct=result['sum_prediction']==gold
        write_json(out/'records'/f'{n:06d}.json',{'episode':e,'result':result,'correct':correct,'smoke_only':smoke})
        key=(e['condition'],e['k']);groups.setdefault(key,[0,0,0,0,0,0,0])
        strict=result['strict_prediction_index'];exact=result['exact_format_prediction_index']
        a=groups[key];a[0]+=correct;a[1]+=1;a[2]+=strict is None;a[3]+=result['sum_prediction']!=result['mean_prediction']
        a[4]+=strict is not None and e['displayed_options'][strict]==gold
        a[5]+=exact is None
        a[6]+=exact is not None and e['displayed_options'][exact]==gold
        if n%20==0:print(f'evaluated {n+1}/{len(rows)}',flush=True)
    return {'cells':[{'condition':c,'k':k,'n':a[1],'accuracy':a[0]/a[1],'strict_invalid':a[2]/a[1],'sum_mean_disagreement':a[3]/a[1],
                     'strict_generation_accuracy':a[4]/a[1],'exact_format_invalid':a[5]/a[1],
                     'exact_format_generation_accuracy':a[6]/a[1]} for(c,k),a in sorted(groups.items())],
            'generation_parser':'shared normalized whole-label grammar; exact format is a separate case/punctuation-sensitive diagnostic'}


def run(config,args):
    require(getattr(args,'mode','evaluate')=='evaluate','Native family probes are frozen evaluation only')
    task,rows=scheduled(config,args);out=Path(args.output)
    require(not out.exists() or not any(out.iterdir()),'Use a fresh output directory')
    require(not args.checkpoint,'Frozen family probes do not use an adapter')
    spec=read_json(source_path(config['spec']))
    engine=load_native(spec,args.model_dir,args.device,family_runtime=runtime_paths(config,args))
    write_json(out/'runtime.json',engine.identity)
    result=evaluate(engine,rows,image_loader(args.data_root,task),out,smoke=args.smoke)
    write_json(out/'summary.json',result)
    write_json(out/'run.json',{'study':'native_probes','model_id':spec['model_id'],'task':task,'seed':args.seed,'n':len(rows),'smoke_only':args.smoke})
    return result
