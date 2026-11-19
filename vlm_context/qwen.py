"""Portable loading around the accepted native Qwen scorer.

The paper's confirmation path loads the base in BF16 and then upcasts the
represented values to FP32. Loading original weights directly in FP32 is a
different numerical experiment and is recorded as such when requested.
"""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys
from .qwen_base import (QwenEvaluator, MODEL_ID, MODEL_REVISION, SAMPLERS, REMAP,
                        render_messages, validate_answer_boundary)
from .explicit import ExplicitEvaluator
from .protocol import prepare_episode, config_hash
from .io import require, sha256, read_json, source_path
from .baseline_scoring import native_control

def episode_for(query, supports, names, k, seed, sampler, condition):
    require(sampler == SAMPLERS[1], 'Public evaluation uses randomized balanced supports without replacement')
    require(condition in ('standard', 'episode_remap'), 'Unknown label condition')
    value = prepare_episode(query, supports, names, k, seed,
        remap_labels=REMAP if condition == 'episode_remap' else None)
    value.update(sampler=sampler, condition=condition, seed=seed, stage='scheduled_evaluation')
    value.pop('model_integration_verified', None)
    value['support_labels'] = [value['label_space'][x['class_id']] for x in value['supports']]
    return value

def resolve_runtime(model_dir, device, load_dtype='bfloat16'):
    path = Path(model_dir).expanduser().resolve()
    require(path.is_dir(), 'Model directory missing; use the fetch-model command first')
    config = read_json(path / 'config.json')
    require(config.get('model_type') == 'qwen2_vl', 'This scorer requires Qwen2-VL; use the native-family runners for other models')
    require(device in ('cpu', 'cuda', 'mps'), 'Unsupported device')
    require(load_dtype in ('float32', 'bfloat16'), 'FP16 is outside the validated scorer contract')
    pinned = read_json(source_path('configs/qwen2_snapshot.json' if config.get('hidden_size') == 1536
                                    else 'configs/qwen2_7b_snapshot.json'))
    for filename, expected in pinned['files'].items():
        require((path / filename).is_file() and sha256(path / filename) == expected,
                'Pinned model snapshot file missing or changed: ' + filename)
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'numpy', 'Pillow', 'accelerate', 'tokenizers', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'model_id': pinned['model_id'],
        'model_revision': pinned['revision'], 'snapshot_sha256': pinned['files'], 'snapshot': str(path), 'device': device,
        'dtype': load_dtype, 'attention_implementation': 'eager',
        'packages': versions, 'python': sys.version, 'platform': platform.platform(),
        'config_sha256': sha256(path / 'config.json'), 'offline': True,
        'model_fallback': False}

def make_evaluator(model_dir, device, seed=0, checkpoint=None,
                   inference_dtype='float32', *, load_dtype='bfloat16', explicit=True):
    runtime = resolve_runtime(model_dir, device, load_dtype)
    cls = ExplicitEvaluator if explicit else QwenEvaluator
    evaluator = cls(runtime, seed=seed)
    if checkpoint is not None:
        from peft import PeftModel
        wrapper = PeftModel.from_pretrained(evaluator.model, str(checkpoint),
            is_trainable=False, local_files_only=True)
        evaluator.model = wrapper.get_base_model()
    require(inference_dtype in ('float32', 'bfloat16'), 'Unsupported evaluation dtype')
    evaluator.model.to(dtype=getattr(evaluator.torch, inference_dtype))
    evaluator.model.eval()
    evaluator.model.config.use_cache = True
    evaluator.identity.update(checkpoint_load_dtype=load_dtype,
        evaluation_dtype=inference_dtype, parameter_dtype_counts=evaluator._dtype_counts(),
        runtime={**runtime, 'dtype': inference_dtype, 'checkpoint_load_dtype': load_dtype}, adapter_checkpoint=str(checkpoint) if checkpoint else None)
    return evaluator
