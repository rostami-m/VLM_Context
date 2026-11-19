"""Editable/source-only build: large evidence assets remain in the checkout."""
from setuptools.build_meta import (build_editable, get_requires_for_build_editable,
    prepare_metadata_for_build_editable, get_requires_for_build_sdist, build_sdist)

def build_wheel(*args, **kwargs):
    raise RuntimeError('This research release requires its source data/config tree. Clone the repository and use pip install -e .; standalone wheels are intentionally unsupported.')

def get_requires_for_build_wheel(config_settings=None):
    return []
