"""Accepted official-image selection, preprocessing and role assignment."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
import hashlib, io, json, os, tempfile
from pathlib import Path
from .protocol import ImageRecord, config_hash, validate_splits
from .io import write_json as immutable_json


VERSION = "revision-official-dataset-roles-1"



TASKS = {
    "cifar4": {"dataset_id": "uoft-cs/cifar10", "revision": "0b2714987fa478483af9968de7c934580d0bb9a2",
               "cache_path": "uoft-cs___cifar10/plain_text/0.0.0", "prefix": "cifar10", "image_column": "img",
               "source_classes": [0, 1, 2, 3], "class_names": ["airplane", "automobile", "bird", "cat"],
               "source_class_names": ["airplane", "automobile", "bird", "cat"],
               "split_sizes": {"train": 50000, "test": 10000}},
    "fashion4": {"dataset_id": "zalando-datasets/fashion_mnist", "revision": "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2",
                 "cache_path": "zalando-datasets___fashion_mnist/fashion_mnist/0.0.0", "prefix": "fashion_mnist",
                 "image_column": "image", "source_classes": [0, 1, 2, 9],
                 "class_names": ["t-shirt", "trouser", "pullover", "ankle boot"],
                 "source_class_names": ["T - shirt / top", "Trouser", "Pullover", "Ankle boot"],
                 "split_sizes": {"train": 60000, "test": 10000}},
}



ROLE_COUNTS = {"adapter_training_support": 64, "adapter_training_query": 64,
               "development": 32, "evaluation_support": 64, "evaluation_query": 10,
               "confirmation_query": 20, "controller_development": 32, "controller_final": 40}



ROLE_PARTITIONS = {role: ("test" if role in {"evaluation_query", "confirmation_query", "controller_final"}
                           else "train") for role in ROLE_COUNTS}



def decoded_hash(image):
    image = image.convert("RGB")
    header = json.dumps(["RGB", image.width, image.height], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes()).hexdigest()



def model_image(image):
    from PIL import Image
    # The historical RGB PIL resize default is bicubic. Declare it explicitly.
    return image.convert("RGB").resize((112, 112), resample=Image.Resampling.BICUBIC)



def _save_image(path, image, expected_hash):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as previous:
            if decoded_hash(previous) != expected_hash:
                raise ValueError(f"existing dataset image changed: {path}")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png", delete=False) as out:
            temporary = Path(out.name)
        image.save(temporary, format="PNG")
        try:
            os.link(temporary, path)
        except FileExistsError:
            with Image.open(path) as previous:
                if decoded_hash(previous) != expected_hash:
                    raise ValueError(f"concurrent dataset image differs: {path}")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)



def _source_id(spec, split, row):
    return f"{spec['dataset_id']}@{spec['revision']}/{split}/{row:06d}"



def _describe(spec, split, row, sources):
    source = sources[split].image(row)
    rendered = model_image(source)
    original_class = sources[split].labels[row]
    record = {"image_id": _source_id(spec, split, row), "source_partition": split,
              "source_row": row, "source_class_id": original_class,
              "class_id": spec["source_classes"].index(original_class),
              "source_mode": source.mode, "source_size": list(source.size),
              "native_rgb_sha256": decoded_hash(source), "content_sha256": decoded_hash(rendered),
              "model_rgb112_sha256": decoded_hash(rendered)}
    return record, rendered



def _historical_rows(sources, classes, train_per_class, test_total):
    selected = {"train": set(), "test": set()}
    for source_class in classes:
        rows = [i for i, label in enumerate(sources["train"].labels) if label == source_class]
        selected["train"].update(rows[:train_per_class])
    # Submitted loader selected first n filtered test rows; recovered loader
    # selected class-balanced first n rows. Reserve both complete pools.
    filtered = [i for i, label in enumerate(sources["test"].labels) if label in classes]
    selected["test"].update(filtered[:test_total])
    for offset, source_class in enumerate(classes):
        count = test_total // len(classes) + int(offset < test_total % len(classes))
        rows = [i for i, label in enumerate(sources["test"].labels) if label == source_class]
        selected["test"].update(rows[:count])
    return selected



def assemble_manifest(task, root, sources, provenance, *, role_counts=None,
                      historical_train_per_class=96, historical_test_total=120):
    """Prepare a new deterministic role bank; fixtures may pass in-memory sources."""
    from PIL import __version__ as pillow_version
    spec, root = TASKS[task], Path(root)
    counts = dict(ROLE_COUNTS if role_counts is None else role_counts)
    if set(counts) != set(ROLE_COUNTS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("all explicit role counts must be positive integers")
    config = {"version": VERSION, "task": task, "dataset_spec": spec, "role_counts_per_class": counts,
              "role_partitions": ROLE_PARTITIONS, "pillow_version": pillow_version,
              "transform": {"convert": "RGB", "resize": [112, 112], "resampling": "PIL.BICUBIC"},
              "historical_exclusion": {"train_first_per_class": historical_train_per_class,
                                       "test_union_first_filtered_and_balanced_total": historical_test_total},
              "provenance": provenance,
              "allocation": "SHA256-ranked source IDs per task/partition/class; fixed role order; no experiment seed"}
    fingerprint = config_hash(config)
    existing = root / "image_manifest.json"
    if existing.exists() and json.loads(existing.read_text())["config_sha256"] != fingerprint:
        raise ValueError("dataset configuration changed; use a new output version")
    historical = _historical_rows(sources, spec["source_classes"], historical_train_per_class, historical_test_total)
    excluded, used_ids, used_native, used_rendered = [], set(), set(), set()
    for split, rows in historical.items():
        for row in sorted(rows):
            record, _ = _describe(spec, split, row, sources)
            excluded.append(record)
            used_ids.add(record["image_id"])
            used_native.add(record["native_rgb_sha256"])
            used_rendered.add(record["content_sha256"])
    role_records, image_records, rejected = {}, {}, []
    # Stable source ordering is independent of run seed, K, scorer and method.
    candidates = {}
    for split in ("train", "test"):
        for c, source_class in enumerate(spec["source_classes"]):
            rows = [i for i, label in enumerate(sources[split].labels) if label == source_class]
            candidates[(split, c)] = sorted(rows, key=lambda row: config_hash(
                [VERSION, task, split, c, _source_id(spec, split, row)]))
    for role in ROLE_COUNTS:
        split = ROLE_PARTITIONS[role]
        role_records[role] = []
        for c in range(4):
            selected = 0
            for row in candidates[(split, c)]:
                identity = _source_id(spec, split, row)
                if identity in used_ids:
                    continue
                record, rendered = _describe(spec, split, row, sources)
                reasons = []
                if record["native_rgb_sha256"] in used_native:
                    reasons.append("native_rgb_duplicate_of_historical_or_previously_reserved_image")
                if record["content_sha256"] in used_rendered:
                    reasons.append("model_rgb112_duplicate_of_historical_or_previously_reserved_image")
                if reasons:
                    rejected.append({"image_id": identity, "reasons": reasons})
                    # Rejected pixels may never enter any later role.
                    used_ids.add(identity)
                    continue
                path = Path("images") / role / f"c{c}_{row:06d}.png"
                record["path"] = str(path)
                _save_image(root / path, rendered, record["content_sha256"])
                image_records[identity] = record
                role_records[role].append(ImageRecord(identity, record["content_sha256"], c))
                used_ids.add(identity)
                used_native.add(record["native_rgb_sha256"])
                used_rendered.add(record["content_sha256"])
                selected += 1
                if selected == counts[role]:
                    break
            if selected != counts[role]:
                raise ValueError(f"insufficient unique nonhistorical {split} images for {role}, class {c}")
    validate_splits(role_records)
    native_records = {role: [ImageRecord(r.image_id, image_records[r.image_id]["native_rgb_sha256"], r.class_id)
                             for r in records] for role, records in role_records.items()}
    validate_splits(native_records)
    manifest = {"generator_version": VERSION, "task": task, "config_sha256": fingerprint,
                "config": config, "class_names": spec["class_names"], "images": image_records,
                "roles": {role: [r.image_id for r in records] for role, records in role_records.items()},
                "decoded_hash_convention": "SHA256(JSON([RGB,width,height]) + NUL + RGB bytes)",
                "content_hash_representation": "model input RGB112 bicubic image stored in PNG",
                "historical_excluded_records": excluded, "duplicate_rejections": rejected,
                "verification": {"roles_disjoint_by_source_id": True, "roles_disjoint_by_native_rgb": True,
                                 "roles_disjoint_by_model_rgb112": True,
                                 "historical_exclusions_checked_in_both_pixel_representations": True},
                "limitations": ["Historical exclusions cover the documented 96-per-class training support and 120-item test pools, including both observed loader orders.",
                                "Unrecorded historical exposures are not established by this manifest; accepted checkpoint reuse still requires its own provenance gate.",
                                "Cached revision paths and file hashes establish these local inputs, not historical runtime use or pretraining absence.",
                                "Prepared image banks contain no model evaluations or fitted checkpoints."]}
    immutable_json(existing, manifest)
    return manifest, role_records
