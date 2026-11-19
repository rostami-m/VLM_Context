"""Instantiate the accepted joint/depth target ledger without changing its ranks."""
from __future__ import annotations
from .io import require
from .joint_targets import validate_instantiated_targets


def attach_adapter(evaluator,expected,dropout):
    from peft import LoraConfig,get_peft_model
    torch=evaluator.torch
    actual={n:(m.in_features,m.out_features) for n,m in evaluator.model.named_modules() if isinstance(m,torch.nn.Linear)}
    for row in expected['modules']:
        require(actual.get(row['name'])==(row['in_features'],row['out_features']),'actual base Linear dimensions differ')
    cfg=LoraConfig(r=min(expected['rank_pattern'].values()),lora_alpha=2*min(expected['rank_pattern'].values()),
        lora_dropout=dropout,target_modules=expected['target_regex'],bias='none',inference_mode=False,
        rank_pattern=expected['rank_pattern'],alpha_pattern=expected['alpha_pattern'])
    adapter=get_peft_model(evaluator.model,cfg);evaluator.model=adapter.get_base_model()
    injected=sorted(n for n,m in evaluator.model.named_modules() if hasattr(m,'lora_A') and hasattr(m,'lora_B'))
    require(injected==sorted(expected['rank_pattern']),'unexpected injected module')
    rows=[]
    for name,tensor in adapter.named_parameters():
        if tensor.requires_grad:
            require(name.startswith('base_model.model.'),'unexpected PEFT tensor prefix')
            rows.append({'name':name[len('base_model.model.'):],'shape':list(tensor.shape),
                         'numel':tensor.numel(),'dtype':str(tensor.dtype)})
    validate_instantiated_targets(expected,rows)
    return adapter,{**expected,'trainable_tensors':rows,'runtime_instantiated':True}
