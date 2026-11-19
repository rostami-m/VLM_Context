"""Shared adapter construction and the accepted query-answer loss contract."""
from .training_primitives import (complete_answer_mask, training_batch,
    parameter_hashes, compare_roundtrip, require_positive_norms, require_error_bound)
from .targets import target_inventory, expected_linear_shapes, arm_specs
from .joint_adaptation import attach_adapter as attach_ledger
from .io import require

def diagnostic_ledger(locus):
    value = target_inventory(expected_linear_shapes(), locus)
    value['rank_pattern'] = value['rank_by_target']
    value['alpha_pattern'] = value['alpha_by_target']
    return value

def fit(evaluator, episodes, image_loader, ledger, recipe, output):
    """Source-derived AdamW loop; fresh output, full-answer loss, FP32 adapters."""
    from pathlib import Path
    import math
    import time
    from .io import write_json, sha256
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), 'Use a fresh fit directory')
    require(episodes, 'Training schedule is empty')
    write_json(output / 'runtime.json', evaluator.identity)
    adapter, actual = attach_ledger(evaluator, ledger, recipe.get('dropout', .05))
    torch = evaluator.torch
    write_json(output / 'target_ledger.json', actual)
    from .protocol import config_hash
    write_json(output / 'training_recipe.json', {'recipe': recipe, 'episode_count': len(episodes),
        'schedule_sha256': config_hash(episodes)})
    frozen_before = {n: h for n, h in parameter_hashes(evaluator.model, torch).items() if '.lora_' not in n}
    write_json(output / 'frozen_base_before.json', frozen_before)
    evaluator.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    evaluator.model.enable_input_require_grads()
    evaluator.model.config.use_cache = False
    adapter.train()
    named = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
    require(all('lora_' in n and str(p.dtype) == 'torch.float32' for n, p in named),
            'Only FP32 LoRA parameters may be trained')
    optimizer = torch.optim.AdamW([p for _, p in named], lr=recipe['lr'],
        betas=tuple(recipe['betas']), eps=recipe['eps'], weight_decay=recipe['weight_decay'])
    losses = []
    started = time.monotonic()
    for step, episode in enumerate(episodes):
        optimizer.zero_grad(set_to_none=True)
        batch, metadata = training_batch(evaluator, episode, image_loader)
        result = adapter(**batch, use_cache=False, return_dict=True)
        loss = result.loss
        require(bool(torch.isfinite(loss).item()), 'Nonfinite loss')
        positions = metadata['supervision']['prediction_positions']
        target = torch.tensor(metadata['supervision']['answer_token_ids'], device=evaluator.device)
        logits = result.logits[0, positions].detach().float()
        manual = torch.nn.functional.cross_entropy(logits, target)
        require(abs(float(loss.item())-float(manual.item())) <= (1e-5 if evaluator.device == 'cpu' else .005),
                'Model loss differs from complete-answer causal loss')
        previous = [p.detach().clone() for _, p in named]
        loss.backward()
        norms = []
        for name, parameter in named:
            require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item()),
                    'Missing/nonfinite adapter gradient: ' + name)
            norms.append(float(parameter.grad.detach().float().norm().item()))
        require_positive_norms(norms, 'adapter gradients')
        require(all(p.grad is None for n, p in adapter.named_parameters() if not p.requires_grad),
                'Frozen base received gradients')
        optimizer.step()
        updates = [float((p.detach() - before).float().norm().item()) for (_, p), before in zip(named, previous)]
        require_positive_norms(updates, 'adapter updates')
        del previous
        require(all(bool(torch.isfinite(p).all().item()) for _, p in named), 'Nonfinite updated adapter')
        require(all(not torch.is_tensor(value) or bool(torch.isfinite(value).all().item())
                    for _, p in named for value in optimizer.state[p].values()), 'Nonfinite AdamW state')
        value = float(loss.item())
        losses.append(value)
        write_json(output / 'steps' / f'{step:04d}.json', {'step': step, 'loss': value,
            'manual_loss': float(manual.item()), 'gradient_norms': norms, 'update_norms': updates, **metadata})
        print(f'optimizer step {step + 1}/{len(episodes)} loss={value:.6f}', flush=True)
        del result, loss, manual, logits, batch, target
    require_positive_norms([float(p.detach().float().norm().item()) for n,p in named if '.lora_B.' in n],
                           'learned LoRA output factors')
    frozen_after = {n: h for n, h in parameter_hashes(evaluator.model, torch).items() if '.lora_' not in n}
    require(frozen_after == frozen_before, 'Frozen base changed during adapter training')
    write_json(output / 'frozen_base_after.json', frozen_after)
    adapter.save_pretrained(output / 'checkpoint', safe_serialization=True)
    evaluator.model.gradient_checkpointing_disable()
    evaluator.model.disable_input_require_grads()
    evaluator.model.config.use_cache = True
    adapter.eval()
    receipt = {'completed_steps': len(episodes), 'trainable_parameters': actual['trainable_parameters'],
        'first_loss': losses[0], 'last_loss': losses[-1], 'mean_loss': math.fsum(losses)/len(losses),
        'seconds': time.monotonic()-started, 'adapter_parameter_sha256': parameter_hashes(evaluator.model, torch, adapter_only=True),
        'frozen_base_unchanged': True, 'checkpoint_files': {p.name: sha256(p) for p in (output/'checkpoint').iterdir() if p.is_file()}}
    write_json(output / 'fit_complete.json', receipt)
    return receipt
