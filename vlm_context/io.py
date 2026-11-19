"""Portable JSON, hashing, and immutable result files."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)

def write_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    if path.exists() and not overwrite:
        require(path.read_bytes() == data, 'Output differs; choose a fresh output directory: ' + str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def package_root():
    return Path(__file__).resolve().parent.parent

def source_path(relative):
    path = (package_root() / relative).resolve()
    require(path.is_relative_to(package_root()), 'Resource path escapes repository')
    require(path.exists(), 'Resource missing. Run from a complete source checkout installed with pip install -e .: ' + str(relative))
    return path

read = read_json
publish = write_json
sha = sha256
