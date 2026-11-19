"""The controller's BF16 training and value-preserving FP32 inference phases."""
import hashlib

from ..continual.native import NativeRun
from ..continual.contract import require, write


def canonical_parameters(model, torch):
    result = {}
    for name, value in model.named_parameters():
        name = name.replace('.base_layer.', '.')
        require(name not in result, 'Canonical parameter-name collision')
        tensor = value.detach().to(device='cpu', dtype=torch.float32).contiguous()
        result[name] = {'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                        'numel': value.numel(), 'dtype': str(value.dtype)}
    return result


class ControllerRun(NativeRun):
    def __init__(self, *args, **kwargs):
        self.precision_index = 0
        super().__init__(*args, **kwargs)

    def change_precision(self, fp32):
        ev, torch = self.evaluator, self.torch
        target = torch.float32 if fp32 else torch.bfloat16
        parameters = dict(ev.model.named_parameters())
        base = {k: v for k, v in parameters.items() if '.lora_' not in k}
        adapters = {k: v for k, v in parameters.items() if '.lora_' in k}
        require(all(v.dtype == torch.float32 for v in adapters.values()), 'LoRA values must remain FP32')
        if all(v.dtype == target for v in base.values()):
            return
        before = canonical_parameters(ev.model, torch)
        buffers = {k: (str(v.dtype), list(v.shape)) for k, v in ev.model.named_buffers()}
        with torch.no_grad():
            for value in base.values():
                value.data = value.data.to(dtype=target)
        ev.synchronize()
        if ev.device == 'mps':
            torch.mps.empty_cache()
        elif str(ev.device).startswith('cuda'):
            torch.cuda.empty_cache()
        after = canonical_parameters(ev.model, torch)
        require(set(before) == set(after) and all(
            (v['numel'], v['sha256']) == (after[k]['numel'], after[k]['sha256']) for k, v in before.items()),
            'Phase conversion changed numerical parameter values')
        require(buffers == {k: (str(v.dtype), list(v.shape)) for k, v in ev.model.named_buffers()}, 'Phase conversion changed buffers')
        self.precision_index += 1
        write(self.output / 'precision' / f'{self.precision_index:04d}.json', {
            'phase': 'inference_FP32' if fp32 else 'training_BF16_base_FP32_LoRA',
            'canonical_before': before, 'canonical_after': after, 'values_unchanged': True})

    def evaluate(self, *args, **kwargs):
        self.change_precision(True)
        result = super().evaluate(*args, **kwargs)
        result['inference_dtype'] = 'float32'
        if not hasattr(self, 'base_hash'):
            self.change_precision(False)
        return result

    def train(self, *args, **kwargs):
        self.change_precision(False)
        return super().train(*args, **kwargs)

    def checkpoint(self, *args, **kwargs):
        self.change_precision(False)
        return super().checkpoint(*args, **kwargs)

    # This version restores training precision before the final frozen-base hash.
    from .execution import execute
