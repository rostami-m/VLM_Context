"""Download or reconstruct exact image banks from portable source manifests."""
from pathlib import Path
from .io import read_json, write_json, require, source_path
from .data_contract import NAMES, ROLES, load_image, dataset_roles
from .dataset_primitives import TASKS, model_image, _save_image
from .shapes import make_manifest, decoded_hash

def load_manifest(data_root, task):
    return read_json(Path(data_root) / task / 'image_manifest.json')

def image_loader(data_root, task):
    root = Path(data_root) / task
    manifest = load_manifest(data_root, task)
    return lambda identifier: load_image(root, manifest, identifier)

def prepare_shapes(data_root):
    root = Path(data_root) / 'shapes4'
    manifest, _ = make_manifest(root)
    accepted = read_json(source_path('data/protocols/datasets/shapes4_manifest.json.gz'))
    require(manifest['images'] == accepted['images'] and manifest['roles'] == accepted['roles'],
            'Generated Shapes pixels/roles differ from the accepted manifest')
    write_json(root / 'image_manifest.json', manifest)
    return manifest

def prepare_from_manifest(task, manifest_path, data_root, *, cache_dir=None):
    """Fetch source rows, then require exact decoded-pixel hashes.

    Only the source datasets are downloaded. No private filesystem paths or
    machine-specific Arrow cache names are required.
    """
    require(task in TASKS, 'Use prepare_shapes for the synthetic image bank')
    from datasets import load_dataset
    spec = TASKS[task]
    manifest = read_json(manifest_path)
    dataset_roles(task, manifest)
    splits = {}
    root = Path(data_root) / task
    for identifier, record in manifest['images'].items():
        split = record.get('source_split') or record.get('official_split')
        row = record.get('source_row')
        if row is None:
            row = record.get('source_row_index')
        # Stable IDs contain the official split and row in accepted manifests.
        source_id = record.get('source_id', identifier)
        if split is None or row is None:
            parts = source_id.split('/')
            require(parts[-2] in ('train', 'test') and parts[-1].isdigit(),
                    'Manifest does not expose an official source row: ' + identifier)
            split, row = parts[-2], int(parts[-1])
        if split not in splits:
            splits[split] = load_dataset(spec['dataset_id'], revision=spec['revision'],
                split=split, cache_dir=cache_dir)
        original = splits[split][int(row)][spec['image_column']]
        image = model_image(original)
        require(decoded_hash(image) == record['content_sha256'],
                'Source image/preprocessing differs from the accepted manifest: ' + identifier)
        _save_image(root / record['path'], image, record['content_sha256'])
    write_json(root / 'image_manifest.json', manifest)
    return manifest
