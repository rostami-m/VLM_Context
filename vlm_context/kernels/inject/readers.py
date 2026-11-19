"""Load source-only Q05 readers with completed-fit and checkpoint checks."""
from contextlib import contextmanager
from pathlib import Path
from vlm_context.io import read_json, sha256, require


def reference(project, path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256(path)}


def verify_ref(project, ref):
    path = Path(ref['path'])
    require(sha256(path) == ref['sha256'], 'Referenced evidence changed')
    return read_json(path)


def completed_reader(session, name, seed):
    require(name in ('connector_episode_remap', 'uncapped_all_episode_remap'), 'Unknown source reader')
    pattern = session.config['reader_checkpoints'][name]
    checkpoint = Path(pattern.format(seed=seed)).expanduser().resolve()
    run = read_json(checkpoint.parent / 'run.json')
    fit = read_json(checkpoint.parent / 'fit_complete.json')
    require(run['study'] == 'q05' and run['mode'] == 'train' and run['task'] == 'shapes4'
            and run['arm'] == name and run['seed'] == seed and run['smoke_only'] is False
            and run['episodes'] == fit['completed_steps'] == 150, 'Inject needs a complete matching Shapes source fit')
    expected = 135168 if name == 'connector_episode_remap' else 9232384
    require(fit['trainable_parameters'] == expected, 'Reader target budget differs')
    actual = {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()}
    require(actual == fit['checkpoint_files'], 'Reader checkpoint files differ from the fit receipt')
    return fit, checkpoint


@contextmanager
def reader(session, name, seed):
    if name == 'frozen':
        yield {'reader': 'frozen', 'checkpoint': None, 'trainable_parameters': 0}
        return
    fit, path = completed_reader(session, name, seed)
    from peft import PeftModel
    from vlm_context.adaptation import parameter_hashes
    adapter = PeftModel.from_pretrained(session.evaluator.model, str(path),
                                       is_trainable=False, local_files_only=True)
    session.evaluator.model = adapter.get_base_model()
    try:
        hashes = parameter_hashes(session.evaluator.model, session.evaluator.torch, adapter_only=True)
        require(hashes == fit['adapter_parameter_sha256'], 'Reloaded adapter tensor hashes differ')
        parameters = [p for n, p in session.evaluator.model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n]
        require(sum(p.numel() for p in parameters) == fit['trainable_parameters'], 'Reader parameter inventory differs')
        require(all(str(p.dtype) == 'torch.float32' for p in parameters), 'Reader adapters must remain FP32')
        session.evaluator.model.eval()
        yield {'reader': name, 'checkpoint': reference(session.project, path.parent / 'fit_complete.json'),
               'trainable_parameters': fit['trainable_parameters'], 'adapter_parameter_sha256': hashes,
               'completed_training_steps': 150}
    finally:
        session.evaluator.model = adapter.unload()
        require(not any('lora_' in n for n, _ in session.evaluator.model.named_parameters()), 'Adapter unload left parameters behind')
        session.evaluator.model.eval()
