"""Deterministic Shapes images and explicit disjoint data roles.

Uses the submitted renderer's distribution, with new role-specific seeds. These
are new revision images, not a claim to recover historical sample identities.
No torch/transformers imports, and no changes to historical task generators.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from .protocol import ImageRecord, config_hash, validate_splits

NAMES = ["circle", "square", "triangle", "cross"]
GENERATOR_VERSION = "revision-shapes-disjoint-1"
ROLE_COUNTS = {"adapter_training_support": 64, "adapter_training_query": 64,
               "development": 32, "evaluation_support": 64,
               "evaluation_query": 10, "confirmation_query": 20,
               "controller_development": 32, "controller_final": 40}


def decoded_hash(image):
    image = image.convert("RGB")
    header = json.dumps(["RGB", image.width, image.height], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes()).hexdigest()


def draw_shape(kind, rng):
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (112, 112), tuple(rng.randint(200, 255) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    color = tuple(rng.randint(0, 120) for _ in range(3))
    x, y, size = rng.randint(20, 50), rng.randint(20, 50), rng.randint(35, 55)
    if kind == 0:
        draw.ellipse([x, y, x + size, y + size], fill=color)
    elif kind == 1:
        draw.rectangle([x, y, x + size, y + size], fill=color)
    elif kind == 2:
        draw.polygon([(x, y + size), (x + size, y + size), (x + size / 2, y)], fill=color)
    else:
        draw.line([x, y, x + size, y + size], fill=color, width=8)
        draw.line([x, y + size, x + size, y], fill=color, width=8)
    return image


def make_manifest(root, role_counts=None):
    """Generate/resume deterministic PNGs, rechecking decoded pixels on resume."""
    from PIL import Image, __version__ as pillow_version
    root = Path(root)
    roles, images = {}, {}
    for role, count in (ROLE_COUNTS if role_counts is None else role_counts).items():
        roles[role] = []
        for class_id in range(4):
            for index in range(count):
                image_id = f"{GENERATOR_VERSION}/{role}/c{class_id}/{index:04d}"
                seed = int(config_hash([GENERATOR_VERSION, image_id]), 16)
                image = draw_shape(class_id, random.Random(seed))
                pixel_hash = decoded_hash(image)
                path = root / "images" / role / f"c{class_id}_{index:04d}.png"
                if path.exists():
                    with Image.open(path) as existing:
                        if decoded_hash(existing) != pixel_hash:
                            raise ValueError(f"image content changed: {path}")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)
                record = ImageRecord(image_id, pixel_hash, class_id)
                roles[role].append(record)
                images[image_id] = {**asdict(record), "path": str(path.relative_to(root))}
    validate_splits(roles)
    manifest = {"generator_version": GENERATOR_VERSION, "pillow_version": pillow_version,
                "decoded_hash_convention": "SHA256(JSON([RGB,width,height]) + NUL + RGB bytes)",
                "class_names": NAMES, "roles": {k: [r.image_id for r in v] for k, v in roles.items()},
                "images": images}
    return manifest, roles


def load_image(root, manifest, image_id):
    from PIL import Image
    record = manifest["images"][image_id]
    with Image.open(Path(root) / record["path"]) as source:
        image = source.convert("RGB")
    if decoded_hash(image) != record["content_sha256"]:
        raise ValueError(f"decoded image changed: {image_id}")
    return image
