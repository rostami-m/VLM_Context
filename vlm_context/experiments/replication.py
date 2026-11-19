"""Q13: native connector/last-four-block adaptation in two model families."""
from pathlib import Path
from ..io import read_json,write_json,source_path,require,sha256
from ..data import image_loader
from ..native_families import load_native
from ..family_training import fit
from .native_probes import evaluate
from ..native_checkpoints import validate_replication


def run(config,args):
    out=Path(args.output);require(not out.exists() or not any(out.iterdir()),'Use a fresh output directory')
    arm=getattr(args,'arm',None) or config.get('arm','frozen')
    require(arm in ('frozen','connector','late'),'Q13 reported arms: frozen,connector,late')
    spec=read_json(source_path(config['spec']))
    training=getattr(args,'mode','evaluate')=='train'
    validate_replication(args.checkpoint,arm=arm,seed=args.seed,spec=spec,smoke=args.smoke,training=training)
    loader=image_loader(args.data_root,'shapes4')
    if training:
        import torch,random
        torch.manual_seed(args.seed);random.seed(args.seed)
    engine=load_native(spec,args.model_dir,args.device,checkpoint=None if training else args.checkpoint)
    write_json(out/'runtime.json',engine.identity)
    if training:
        rows=read_json(source_path('data/protocols/replication/training.json.gz'))
        checkpoint=fit(engine,rows,loader,arm,out,args.seed,args.smoke)
        result={'checkpoint':str(checkpoint),'steps':2 if args.smoke else 150}
    else:
        rows=[e for e in read_json(source_path('data/protocols/replication/episodes.json.gz')) if e['seed']==args.seed]
        if args.smoke:rows=rows[:2]
        result=evaluate(engine,rows,loader,out,smoke=args.smoke)
    write_json(out/'summary.json',result)
    write_json(out/'run.json',{'study':'replication','arm':arm,'seed':args.seed,'model_id':spec['model_id'],'model_revision':spec['revision'],'episodes':(2 if args.smoke else 150) if training else len(rows),'smoke_only':args.smoke,'mode':args.mode,'objective':'75 standard + 75 remap complete-answer CE','base_dtype':spec['dtype']})
    if training:
        write_json(out/'fit_complete.json',{'completed_steps':2 if args.smoke else 150,
            'frozen_base_unchanged':True,'reload_verified':True,
            'checkpoint_files':{p.name:sha256(p) for p in checkpoint.iterdir() if p.is_file()},
            'evidence_files':{name:sha256(out/name) for name in ['run.json','runtime.json','recipe.json','reload.json','base_unchanged.json']}})
    return result
