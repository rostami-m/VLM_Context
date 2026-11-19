"""Fail-closed checkpoint lineage checks without importing model libraries."""
from pathlib import Path
from .io import read_json,require,sha256


def checkpoint_files(checkpoint, receipt):
    checkpoint=Path(checkpoint)
    require(checkpoint.is_dir(),'Checkpoint directory is missing')
    actual={p.name:sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    require({'adapter_config.json','adapter_model.safetensors'} <= set(actual),'Complete serialized LoRA adapter is required')
    require(actual==receipt.get('checkpoint_files'),'Checkpoint bytes differ from the completed fit receipt')


def evidence_files(root, receipt):
    inventory=receipt.get('evidence_files')
    require(isinstance(inventory,dict) and inventory,'Completed fit must bind its source metadata')
    for name,digest in inventory.items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts,'Unsafe fit evidence path')
        require((root/path).is_file() and sha256(root/path)==digest,'Fit source metadata changed: '+name)


def validate_replication(checkpoint,*,arm,seed,spec,smoke=False,training=False):
    require(arm in ('frozen','connector','late'),'Unknown Q13 arm')
    if training:
        require(arm!='frozen','Q13 frozen arm cannot be trained')
        require(not checkpoint,'Q13 training starts from the pinned base; do not supply --checkpoint')
        return None
    if arm=='frozen':
        require(not checkpoint,'Frozen Q13 evaluation cannot load an adapter')
        return None
    require(checkpoint,'Adapted Q13 evaluation requires a completed matching fit')
    checkpoint=Path(checkpoint);root=checkpoint.parent
    run=read_json(root/'run.json');receipt=read_json(root/'fit_complete.json')
    require(run.get('study')=='replication' and run.get('mode')=='train' and run.get('arm')==arm and run.get('seed')==seed,'Q13 source study/arm/seed differs')
    require(run.get('model_id')==spec['model_id'] and run.get('model_revision')==spec['revision'],'Q13 source model or revision differs')
    source_smoke=run.get('smoke_only')
    require(type(source_smoke) is bool and (not source_smoke or smoke),'Smoke-trained adapter requires --smoke evaluation')
    expected=2 if source_smoke else 150
    require(run.get('episodes')==receipt.get('completed_steps')==expected,'Q13 fit is incomplete')
    require(receipt.get('frozen_base_unchanged') is True and receipt.get('reload_verified') is True,'Q13 fit lacks frozen-base/reload verification')
    checkpoint_files(checkpoint,receipt);evidence_files(root,receipt)
    require({'run.json','runtime.json','recipe.json','reload.json','base_unchanged.json'} <= set(receipt['evidence_files']),'Q13 source lineage inventory is incomplete')
    recipe=read_json(root/'recipe.json');runtime=read_json(root/'runtime.json')
    require(recipe.get('arm')==arm and recipe.get('seed')==seed and recipe.get('steps')==expected and recipe.get('smoke')==source_smoke,'Q13 fit recipe differs from source run')
    require(runtime.get('model_id')==spec['model_id'] and runtime.get('revision')==spec['revision'],'Q13 recorded runtime differs')
    require(all(recipe.get(k)==v for k,v in {'rank':8,'alpha':16,'dropout':0,'weight_decay':0,'learning_rate':1e-4,'gradient_clip':1}.items()),'Q13 optimizer/adapter recipe differs')
    return receipt


def validate_q11(checkpoint,*,arm,seed,spec,smoke=False):
    require(arm in ('frozen','joint'),'Q11 accepts frozen or Shapes joint arms')
    if arm=='frozen':
        require(not checkpoint,'Frozen Q11 evaluation cannot load an adapter')
        return None
    require(checkpoint,'Q11 joint transfer requires a completed Shapes source checkpoint')
    checkpoint=Path(checkpoint);root=checkpoint.parent
    run=read_json(root/'run.json');receipt=read_json(root/'fit_complete.json')
    require(run.get('study')=='core' and run.get('mode')=='train' and run.get('task')=='shapes4' and run.get('arm')=='joint_uncapped_r16_episode_remap' and run.get('seed')==seed,'Q11 source must be the matching Shapes core joint fit')
    source_smoke=run.get('smoke_only')
    require(type(source_smoke) is bool and (not source_smoke or smoke),'Smoke-trained adapter requires --smoke evaluation')
    require(run.get('episodes')==receipt.get('completed_steps')==(2 if source_smoke else 250),'Q11 source fit is incomplete')
    require(receipt.get('frozen_base_unchanged') is True and receipt.get('trainable_parameters')==12800000,'Q11 source fit target budget or frozen-base verification differs')
    checkpoint_files(checkpoint,receipt)
    runtime=read_json(root/'runtime.json').get('runtime',{})
    require(runtime.get('model_id')==spec['model_id'] and runtime.get('model_revision')==spec['revision'],'Q11 source model or revision differs')
    if 'evidence_files' in receipt:evidence_files(root,receipt)
    return receipt
