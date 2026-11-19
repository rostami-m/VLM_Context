"""Reconstruct the exact CIFAR-100 image bank from pinned official Parquet files."""
from pathlib import Path
import hashlib
import io
import json

from vlm_context.io import read_json, require, sha256, source_path, write_json

DATASET = 'uoft-cs/cifar100'
REVISION = 'aadb3af77e9048adbea6b47c21a81e47dd092ae5'
FILES = {
    'train': ('cifar100/train-00000-of-00001.parquet', '694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23'),
    'test': ('cifar100/test-00000-of-00001.parquet', '98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68'),
}


def decoded_hash(image):
    prefix = json.dumps([image.mode, image.width, image.height], separators=(',', ':')).encode()
    return hashlib.sha256(prefix + b'\0' + image.tobytes()).hexdigest()


def materialize(data_root, parquet_root=None, *, cache_dir=None):
    """Fetch if needed, reconstruct all declared roles, and verify every image.

    The manifest includes reserved exclusion images as well as actual training,
    development and evaluation roles. Runners enforce access to their own roles.
    No image is selected again according to the new machine's iteration order.
    """
    from PIL import Image
    import pyarrow.parquet as pq

    target = Path(data_root) / 'continual'
    manifest = read_json(source_path('data/protocols/continual/image_manifest.json.gz'))
    for split, (remote, expected) in FILES.items():
        if parquet_root is None:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(DATASET, remote, repo_type='dataset', revision=REVISION, cache_dir=cache_dir))
        else:
            source = Path(parquet_root) / Path(remote).name
        require(sha256(source) == expected, 'CIFAR-100 source differs from the pinned official file')
        selected = {r['source_index']: r for r in manifest['images'].values() if r['split'] == split}
        cursor, completed = 0, set()
        for batch in pq.ParquetFile(source).iter_batches(batch_size=128, columns=['img', 'fine_label'], use_threads=False):
            for row in batch.to_pylist():
                index = cursor
                cursor += 1
                if index not in selected:
                    continue
                record = selected[index]
                require(row['fine_label'] == record['fine_class_id'], 'Source class changed')
                with Image.open(io.BytesIO(row['img']['bytes'])) as opened:
                    original = opened.convert('RGB')
                require(decoded_hash(original) == record['raw_rgb_sha256'], 'Source pixels changed')
                image = original.resize((112, 112), Image.Resampling.BICUBIC)
                require(decoded_hash(image) == record['content_sha256'], 'Image transform differs from the paper')
                destination = target / record['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with destination.open('xb') as handle:
                        image.save(handle, format='PNG', compress_level=6)
                require(sha256(destination) == record['file_sha256'], 'PNG encoding differs; use the recorded Pillow environment')
                completed.add(index)
        require(completed == set(selected), 'Not all declared CIFAR-100 rows were reconstructed')
    write_json(target / 'image_manifest.json', manifest)
    return {'images': len(manifest['images']), 'destination': str(target), 'pixel_hashes_verified': True}
