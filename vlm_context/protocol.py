"""Deterministic episode construction and explicit scorer diagnostics.

Standard library only. No images, processors, or models are loaded here. Concepts
selectively adapted from the original incontext_vlm protocol.py, scoring.py,
results.py, and option_order_eval.py. This version separates every RNG stream,
rejects identity leakage/replacement, and gives invalid answers a confusion column.
Model runners separately validate image hashes, token boundaries and EOS policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile


PROTOCOL_VERSION = "revision-preparation-1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _json_bytes(value):
    def validate(item):
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("configuration keys must be strings")
            for child in item.values():
                validate(child)
        elif isinstance(item, list):
            for child in item:
                validate(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("configuration must contain only JSON values")
    validate(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def config_hash(config):
    return hashlib.sha256(_json_bytes(config)).hexdigest()


def _rng(seed, namespace, identity=None):
    _integer(seed, "seed")
    digest = config_hash([PROTOCOL_VERSION, namespace, seed, identity])
    return random.Random(int(digest, 16))


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    content_sha256: str
    class_id: int

    def __post_init__(self):
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be a nonempty stable source identity")
        if not isinstance(self.content_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", self.content_sha256) is None:
            raise ValueError("content_sha256 must identify canonical decoded pixels")
        _integer(self.class_id, "class_id")


def validate_splits(splits):
    """Reject repeated IDs OR content hashes, within and across explicit roles.

    Callers must derive content hashes from canonical decoded pixels, including
    dimensions/mode; this validator cannot verify an asserted hash from pixels.
    """
    if not splits:
        raise ValueError("at least one named split is required")
    seen_ids, seen_hashes = {}, {}
    for role, records in splits.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError("split roles must be named")
        for record in records:
            if not isinstance(record, ImageRecord):
                raise ValueError("split entries must be ImageRecord instances")
            for value, seen, kind in ((record.image_id, seen_ids, "image ID"),
                                      (record.content_sha256, seen_hashes, "content hash")):
                if value in seen:
                    raise ValueError(f"duplicate {kind} across/within {seen[value]} and {role}")
                seen[value] = role


def _balanced(records, n, rng, classes):
    _integer(n, "sample size")
    groups = {c: [] for c in classes}
    for record in sorted(records, key=lambda item: (item.class_id, item.image_id)):
        if record.class_id not in groups:
            raise ValueError("image class is outside the declared label space")
        groups[record.class_id].append(record)
    if not classes:
        if n:
            raise ValueError("cannot sample a nonempty set without classes")
        return ()
    if n and any(not images for images in groups.values()):
        raise ValueError("every declared class needs images for balanced sampling")
    base, remainder = divmod(n, len(classes))
    extra = set(rng.sample(classes, remainder))
    selected = []
    for c in classes:
        count = base + (c in extra)
        if count > len(groups[c]):
            raise ValueError(f"class {c} needs {count} unique images; replacement is disabled")
        selected.extend(rng.sample(groups[c], count))
    rng.shuffle(selected)
    return tuple(selected)


def select_queries(records, n, seed, *, n_classes):
    """Select balanced queries independently of K, condition, or method."""
    _integer(n_classes, "n_classes", 1)
    records = tuple(records)
    validate_splits({"evaluation_query": records})
    return _balanced(records, n, _rng(seed, "queries"),
                     list(range(n_classes)))


def _labels(labels):
    labels = tuple(labels)
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be nonempty strings")
    normalized = tuple(" ".join(label.casefold().split()) for label in labels)
    if len(set(normalized)) != len(normalized):
        raise ValueError("normalized labels must be unique")
    return normalized


def prepare_episode(query, support_pool, names, k, seed, *, remap_labels=None,
                    mapping_seed=None, option_seed=None):
    """Prepare identities only, pairing supports across mapping/option variants.

    The caller must additionally validate all global data roles with
    validate_splits; this function checks the selected query against its pool.
    No replacement policy is offered by this first preparation version.
    """
    names, support_pool = tuple(names), tuple(support_pool)
    _labels(names)
    _integer(k, "k")
    validate_splits({"evaluation_query": (query,), "evaluation_support": support_pool})
    if query.class_id >= len(names):
        raise ValueError("query class is outside the declared label space")
    support_identity = {"query_id": query.image_id, "k": k}
    contrast_identity = {"query_id": query.image_id}
    supports = _balanced(support_pool, k, _rng(seed, "supports", support_identity),
                         list(range(len(names))))
    mapping_seed = seed if mapping_seed is None else mapping_seed
    option_seed = seed if option_seed is None else option_seed
    # Hold vocabulary mapping and displayed order fixed within each query/seed
    # K contrast, including the paired K=0 reference.
    mapping_rng = _rng(mapping_seed, "mapping", contrast_identity)
    option_rng = _rng(option_seed, "options", contrast_identity)
    labels = list(names if remap_labels is None else remap_labels)
    _labels(labels)
    if len(labels) != len(names):
        raise ValueError("remap vocabulary must contain exactly one label per class")
    if remap_labels is not None:
        mapping_rng.shuffle(labels)
    order = list(range(len(names)))
    option_rng.shuffle(order)
    return {
        "protocol_version": PROTOCOL_VERSION, "stage": "preparation_only",
        "model_integration_verified": False, "k": k, "query": asdict(query),
        "supports": [asdict(record) for record in supports],
        "seed_roles": {"support": seed, "mapping": mapping_seed, "options": option_seed},
        "label_space": labels, "option_order": order,
        "displayed_options": [labels[index] for index in order],
    }


def legacy_parse(answer, label_space):
    """Exactly reproduce the historical earliest-substring/tie-by-index parser."""
    label_space = tuple(label_space)
    _labels(label_space)
    lowered = answer.lower()
    hits = [(lowered.find(label.lower()), index) for index, label in enumerate(label_space)
            if label.lower() in lowered]
    return min(hits)[1] if hits else None


def parse_diagnostics(answer, label_space, *, natural_labels=None):
    """Strict grammar: one normalized whole label, with optional trailing .!? marks.

    Whole-word mentions are diagnostics only: verbose/negative text never becomes
    a strict prediction. Exact multiword labels take precedence over overlapping
    shorter labels. Natural-label emissions count only labels outside the options.
    """
    label_space = tuple(label_space)
    labels = _labels(label_space)
    normalized = " ".join(answer.casefold().split())
    def mentions(label):
        return re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", normalized) is not None
    hits = [i for i, label in enumerate(labels) if mentions(label)]
    exact = normalized if normalized in labels else normalized.rstrip(".!?").rstrip()
    prediction = labels.index(exact) if exact in labels else None
    if prediction is not None:
        status = "exact"
    elif len(hits) == len(labels) and len(hits) > 1:
        status = "option_echo"
    elif len(hits) > 1:
        status = "ambiguous_labels"
    else:
        status = "invalid_format" if hits else "no_label"
    natural_hits = ([] if natural_labels is None else
                    [i for i, label in enumerate(_labels(natural_labels))
                     if label not in labels and mentions(label)])
    return {"prediction": prediction, "status": status, "matched_classes": hits,
            "legacy_prediction": legacy_parse(answer, label_space),
            "natural_label_matches": natural_hits, "natural_label_emission": bool(natural_hits)}


def summarize_predictions(records, n_classes):
    """Accuracy retains invalid answers; macro accuracy covers observed gold classes."""
    _integer(n_classes, "n_classes", 1)
    matrix = [[0] * (n_classes + 1) for _ in range(n_classes)]
    for record in records:
        gold, prediction = record["gold"], record["prediction"]
        _integer(gold, "gold")
        if prediction is not None:
            _integer(prediction, "prediction")
        if gold >= n_classes or (prediction is not None and prediction >= n_classes):
            raise ValueError("prediction/gold outside label space")
        matrix[gold][n_classes if prediction is None else prediction] += 1
    counts = [sum(row) for row in matrix]
    n, correct = sum(counts), sum(matrix[c][c] for c in range(n_classes))
    class_accuracy = [matrix[c][c] / count for c, count in enumerate(counts) if count]
    return {"n": n, "accuracy": correct / n if n else None,
            "macro_accuracy": math.fsum(class_accuracy) / len(class_accuracy) if class_accuracy else None,
            "missing_gold_classes": [c for c, count in enumerate(counts) if not count],
            "confusion": matrix, "invalid_column": n_classes,
            "invalid_count": sum(row[-1] for row in matrix)}


def sequence_likelihoods(token_log_probabilities):
    """Reduce already-aligned full-answer log probabilities; does NOT score models.

    Sum is the primary sequence rule; mean is a length-normalized diagnostic.
    Callers must validate answer tokens, causal shifts, shared prefix, and EOS.
    Exact ties choose the first class index; raw scores retain ties for auditing.
    """
    sums, means = [], []
    for values in token_log_probabilities:
        values = list(values)
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) or v > 0 for v in values):
            raise ValueError("each label needs finite nonpositive token log probabilities")
        sums.append(math.fsum(values))
        means.append(sums[-1] / len(values))
    if not sums:
        raise ValueError("at least one candidate label is required")
    return {"sum": sums, "mean": means, "sum_prediction": max(range(len(sums)), key=sums.__getitem__),
            "mean_prediction": max(range(len(means)), key=means.__getitem__)}


def ensure_preparation_manifest(path, config):
    """Atomically create a preparation manifest, or reject any changed identity."""
    path = Path(path)
    payload = {"protocol_version": PROTOCOL_VERSION, "stage": "preparation_only",
               "config_sha256": config_hash(config), "config": config}
    serialized = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)  # Atomic exclusive publication: never overwrite.
        except FileExistsError:
            if _json_bytes(json.loads(path.read_text(encoding="utf-8"))) != serialized:
                raise ValueError("preparation configuration mismatch; use a new output path")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload
