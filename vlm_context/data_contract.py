"""Image identities, fixed role splits, and decoded-pixel validation."""
from __future__ import annotations
from pathlib import Path
from collections import Counter
from .io import require
from .protocol import ImageRecord, validate_splits
from .shapes import decoded_hash


NAMES = {'shapes4': ['circle', 'square', 'triangle', 'cross'],
         'cifar4': ['airplane', 'automobile', 'bird', 'cat'],
         'fashion4': ['t-shirt', 'trouser', 'pullover', 'ankle boot']}



ROLES = ('adapter_training_support', 'adapter_training_query', 'development', 'evaluation_support',
         'evaluation_query', 'confirmation_query', 'controller_development', 'controller_final')



def load_image(root, manifest, image_id):
    from PIL import Image
    record = manifest['images'][image_id]
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    require(path.is_relative_to(root), 'image path escapes dataset root')
    with Image.open(path) as original:
        image = original.convert('RGB')
    require(image.size == (112, 112) and decoded_hash(image) == record['content_sha256'], 'stored model image hash/size changed')
    return image



def dataset_roles(task, manifest):
    require(manifest['class_names'] == NAMES[task] and set(manifest['roles']) == set(ROLES), 'wrong classes or incomplete explicit roles')
    roles = {}
    referenced = []
    native_hashes = set()
    for role, ids in manifest['roles'].items():
        roles[role] = []
        for identifier in ids:
            record = manifest['images'][identifier]
            require(record['image_id'] == identifier and record['class_id'] in range(4), 'invalid image identity/class')
            roles[role].append(ImageRecord(identifier, record['content_sha256'], record['class_id']))
            referenced.append(identifier)
            native = record.get('native_rgb_sha256', record['content_sha256'])
            require(native not in native_hashes, 'duplicate native decoded image across roles')
            native_hashes.add(native)
            if task != 'shapes4':
                expected = 'test' if role in {'evaluation_query', 'confirmation_query', 'controller_final'} else 'train'
                require(record['source_partition'] == expected and record['model_rgb112_sha256'] == record['content_sha256'], 'official partition or model transform mismatch')
        require(set(r.class_id for r in roles[role]) == set(range(4)), 'each role needs all four classes')
    require(set(referenced) == set(manifest['images']) and len(referenced) == len(set(referenced)), 'unassigned/duplicate image identity')
    validate_splits(roles)
    for role, count in [('confirmation_query', 20), ('evaluation_query', 10)]:
        require(Counter(r.class_id for r in roles[role]) == Counter({c: count for c in range(4)}), 'wrong confirmation/pilot balance')
    return roles
