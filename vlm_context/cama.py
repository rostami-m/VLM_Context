"""Isolated candidate-independent CAMA classification adapter (not model-validated).

Reference: https://arxiv.org/html/2505.17097v4, equations 3--10 and the
classification paragraph following Table 2. This is an independent local
implementation, not author code. Demonstrations use ``A: label.`` and the query
uses ``A:``. The single gain uses the first/last A-segment token; those endpoints
are a disclosed interpretation. An extra unmodified prompt probe sums gains
before modulation. Stage-I query weight is zero, matching the literal position
factor. These choices must accompany results, costs, and a matched off control.

No torch/transformers imports or model loads occur at import time. A root-owned
QwenEvaluator can be wrapped with ``CamaAdapter(evaluator)``. The frozen E1
evaluator is never edited. Native eager attention receives finite causal
prefix-only biases through temporary hooks. Unsupported architectures fail.

Verification replays the *recorded same prefix intervention* in no-cache full
and stepwise forwards. Ordinary unmodified candidate-appended forwards are not
valid references. All branches include every label token followed by one EOS.
No candidate or hidden query answer reaches the selector. CPU fixture tests
validate orchestration; an actual pinned-model smoke remains mandatory.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time


VERSION = "revision-cama-prefix-1"
PAPER = "https://arxiv.org/html/2505.17097v4"
DISCLOSURE = {
    "identity": "independent paper implementation with declared local classification/probe choices",
    "prompt": "classification: A: label. demonstration; A: query; common options before all images",
    "gain_endpoints": "first and last A-segment token; endpoints not specified explicitly for classification in paper",
    "selection": "extra unmodified prompt-only probe; gains summed over Stage I before modulated prefill",
    "routing": "modified final Stage-I hidden states; raw-logit head ranking",
    "query_weight": "zero: literal equation-7 positional factor; not the prose-enhancement sensitivity",
    "continuation": "prefill-only intervention; independent candidate caches; full label plus im_end",
    "reference": "same frozen prefix biases replayed causally; appended candidate rows receive no edits",
    "control": "identical prompt, processor, native eager backend, token scoring and greedy limits; modulation disabled",
    "official_code_equivalence_claimed": False,
    "historical_results_equivalent": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class CamaConfig:
    stage1_layers: tuple[int, ...] = (1, 2)
    stage2_layers: tuple[int, ...] = (6, 8, 10, 12, 14, 16, 18)
    token_fraction: float = 0.2
    head_fraction: float = 0.2
    epsilon: float = 1e-8
    stage1_scale: float = 1.0
    stage2_scale: float = 1.0
    max_prefix_tokens: int = 2048
    max_label_and_eos_tokens: int = 16
    max_generated_tokens: int = 16
    max_transient_estimate_bytes: int = 2 * 1024 ** 3

    def __post_init__(self):
        check(bool(self.stage1_layers) and bool(self.stage2_layers), "both stages are required")
        for layers in (self.stage1_layers, self.stage2_layers):
            check(all(type(v) is int and v >= 0 for v in layers) and tuple(sorted(set(layers))) == layers,
                  "layer lists must be sorted unique nonnegative integers")
        check(max(self.stage1_layers) < min(self.stage2_layers), "Stage I must precede Stage II")
        check(all(finite(v) and 0 < v <= 1 for v in (self.token_fraction, self.head_fraction)), "invalid selection fraction")
        check(finite(self.epsilon) and self.epsilon > 0, "epsilon must be positive")
        check(all(finite(v) and 0 <= v <= 1 for v in (self.stage1_scale, self.stage2_scale)), "invalid intervention scale")
        for value in (self.max_prefix_tokens, self.max_label_and_eos_tokens, self.max_generated_tokens,
                      self.max_transient_estimate_bytes):
            check(type(value) is int and value > 0, "limits must be positive integers")


def stable_top(values, fraction):
    check(bool(values) and all(finite(value) for value in values), "selector values must be nonempty and finite")
    check(finite(fraction) and 0 < fraction <= 1, "invalid selection fraction")
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:max(1, math.ceil(len(values) * fraction))]


def softmax(values):
    check(bool(values) and all(finite(value) for value in values), "invalid softmax values")
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(exps)
    return [value / denominator for value in exps]


def single_gain(anchor_mean_logits, epsilon=1e-8):
    """Classification's one positive log-ratio increment; mean heads precedes softmax."""
    check(finite(epsilon) and epsilon > 0, "invalid gain epsilon")
    check(len(anchor_mean_logits) == 2 and len(anchor_mean_logits[0]) == len(anchor_mean_logits[1]),
          "classification requires two anchor rows over the same image tokens")
    before, after = [[max(epsilon, value) for value in softmax(row)] for row in anchor_mean_logits]
    return [max(0.0, right - left) * (math.log(right) - math.log(left)) for left, right in zip(before, after)]


def classification_messages(episode, loader):
    """Only displayed support labels and options enter the prompt; never query class."""
    labels, order = episode["label_space"], episode["option_order"]
    check(len(labels) == 4 and len(set(labels)) == 4 and sorted(order) == list(range(4)), "invalid label/options contract")
    options = [labels[index] for index in order]
    check(episode["displayed_options"] == options, "displayed options differ from permutation")
    check(len(episode["supports"]) == episode["k"], "support count differs from K")
    content = [{"type": "text", "text": "Classify the last image. Answer with exactly one of [" +
                ", ".join(options) + "] and nothing else.\n"}]
    images = []
    for support in episode["supports"]:
        image = loader(support["image_id"])
        images.append(image)
        content.extend([{"type": "image", "image": image},
                        {"type": "text", "text": f"A: {labels[support['class_id']]}."}])
    image = loader(episode["query"]["image_id"])
    images.append(image)
    content.extend([{"type": "image", "image": image}, {"type": "text", "text": "A:"}])
    return [{"role": "user", "content": content}], images


def classification_segments(ids, tokenizer, image_token_id):
    """Exact image/A-segment token boundaries; ambiguous retokenization fails closed."""
    vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    check(len({vision_start, vision_end, im_end, image_token_id}) == 4, "special token identities overlap")
    ranges, index = [], 0
    while index < len(ids):
        if ids[index] != image_token_id:
            index += 1
            continue
        start = index
        while index < len(ids) and ids[index] == image_token_id:
            index += 1
        check(start > 0 and ids[start - 1] == vision_start and index < len(ids) and ids[index] == vision_end,
              "image run lacks exact vision delimiters")
        ranges.append((start, index))
    check(bool(ranges), "CAMA requires images")
    segments = []
    for position, (start, end) in enumerate(ranges):
        text_start, text_end = end + 1, end + 1
        while text_end < len(ids) and ids[text_end] not in (vision_start, im_end):
            text_end += 1
        tokens = ids[text_start:text_end]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if getattr(tokenizer, "is_fast", False):
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
        else:
            pieces = [tokenizer.decode([token], skip_special_tokens=False) for token in tokens]
            check("".join(pieces) == text, "slow tokenizer cannot identify exact anchor offsets")
            offsets, offset = [], 0
            for piece in pieces:
                offsets.append((offset, offset + len(piece)))
                offset += len(piece)
            encoded = tokenizer(text, add_special_tokens=False)
        check(list(encoded["input_ids"]) == tokens, "paired text does not round-trip")
        first, last = len(text) - len(text.lstrip()), len(text.rstrip()) - 1
        check(text[first:].startswith("A:"), "only declared A-only classification prompt is supported")
        query = position == len(ranges) - 1
        check(text[first:].strip() == "A:" if query else text.rstrip().endswith(".") and bool(text[first + 2:-1].strip()),
              "query has an answer or demonstration lacks its displayed label")
        def locate(character):
            matches = [text_start + i for i, (left, right) in enumerate(offsets) if left <= character < right]
            check(len(matches) == 1, "anchor does not have an exact token")
            return matches[0]
        anchors = [locate(first), locate(last)]
        check(anchors[0] < anchors[1], "classification anchors must be distinct causal tokens")
        segments.append({"image": list(range(start, end)), "text": list(range(anchors[0], anchors[1] + 1)), "anchors": anchors})
    return segments


def validate_candidates(candidates, eos, maximum):
    check(type(eos) is int and eos >= 0 and len(candidates) == 4, "four complete candidates and valid EOS required")
    labels = [row["label"] for row in candidates]
    check(all(isinstance(label, str) and label.strip() for label in labels) and len(set(labels)) == 4, "invalid candidate labels")
    for row in candidates:
        tokens = row["label_token_ids"]
        check(bool(tokens) and all(type(token) is int and token >= 0 and token != eos for token in tokens), "empty/invalid label span")
        check(row["scored_token_ids"] == tokens + [eos] and len(tokens) + 1 <= maximum, "complete label/EOS span missing or over budget")


def likelihoods(rows):
    check(len(rows) == 4 and all(row and all(finite(value) and value <= 1e-6 for value in row) for row in rows), "nonfinite/incomplete label scores")
    sums = [math.fsum(row) for row in rows]
    means = [value / len(row) for value, row in zip(sums, rows)]
    return {"sum": sums, "mean": means, "sum_prediction": max(range(4), key=sums.__getitem__),
            "mean_prediction": max(range(4), key=means.__getitem__)}


class SharedPrefixScorer:
    """Backend-neutral branch/replay contract, testable without model libraries.

    backend.prefill receives only prefix data and an enabled flag, never labels.
    The resulting state contains ``first`` and a frozen JSON ``plan``. clone/
    advance operate on one disposable branch. reference replays that plan in
    full/stepwise mode. release drops tensor/cache ownership before references.
    """
    def __init__(self, backend, config=None):
        self.backend, self.config = backend, config or CamaConfig()

    def score(self, prefix, candidates, eos, stop_ids, max_new_tokens, enabled=True, verify=False):
        validate_candidates(candidates, eos, self.config.max_label_and_eos_tokens)
        check(type(max_new_tokens) is int and 0 < max_new_tokens <= self.config.max_generated_tokens, "generation budget exceeded")
        check(type(enabled) is bool and type(verify) is bool, "enabled/verify must be booleans")
        state = self.backend.prefill(prefix, enabled)
        plan = copy.deepcopy(state["plan"])
        identity = digest(plan)
        rows, generated, generated_logps = [], [], []
        try:
            for candidate in candidates:
                branch = self.backend.clone(state)
                try:
                    next_scores, values = state["first"], []
                    for index, token in enumerate(candidate["scored_token_ids"]):
                        values.append(self.backend.value(next_scores, token))
                        if index + 1 < len(candidate["scored_token_ids"]):
                            next_scores = self.backend.advance(branch, token)
                    rows.append(values)
                finally:
                    self.backend.release(branch)
            branch = self.backend.clone(state)
            try:
                next_scores = state["first"]
                for _ in range(max_new_tokens):
                    token = self.backend.argmax(next_scores)
                    generated.append(token)
                    generated_logps.append(self.backend.value(next_scores, token))
                    if token in stop_ids:
                        break
                    if len(generated) < max_new_tokens:
                        next_scores = self.backend.advance(branch, token)
            finally:
                self.backend.release(branch)
            check(digest(state["plan"]) == identity, "candidate continuation mutated frozen intervention")
        finally:
            self.backend.release(state)
        scores = likelihoods(rows)
        validation = None
        if verify:
            full = [self.backend.reference(prefix, row["scored_token_ids"], plan, "full") for row in candidates]
            stepwise = [self.backend.reference(prefix, row["scored_token_ids"], plan, "stepwise") for row in candidates]
            errors = []
            for reference in (full, stepwise):
                likelihoods(reference)
                check(all(len(left) == len(right) for left, right in zip(rows, reference)), "replay score span differs")
                errors.extend(abs(a - b) for left, right in zip(rows, reference) for a, b in zip(left, right))
                check(likelihoods(reference)["sum_prediction"] == scores["sum_prediction"], "cached/replayed primary argmax differs")
            tolerance = self.backend.absolute_tolerance
            check(finite(tolerance) and tolerance > 0 and errors and max(errors) <= tolerance, "cached/replayed scoring tolerance failed")
            replay_generated = self.backend.reference_greedy(prefix, plan, stop_ids, max_new_tokens)
            check(replay_generated == generated, "cached/replayed greedy token sequence differs")
            check(digest(plan) == identity, "reference scoring mutated intervention")
            validation = {"reference_type": "same frozen prefix intervention; no edits to candidate rows",
                          "full_forward_token_logps": full, "independent_stepwise_token_logps": stepwise,
                          "maximum_absolute_logp_difference": max(errors), "absolute_tolerance": tolerance,
                          "generation_matches_no_cache_replay": True, "intervention_sha256": identity}
        return {"per_token_label_log_probabilities": rows, "scores": scores, "greedy_token_ids": generated,
                "greedy_token_log_probabilities": generated_logps, "intervention": plan,
                "intervention_sha256": identity, "independent_validation": validation}


def edit_positions(edit, prefix_length):
    """Sparse causal cells; rows belonging to appended candidates never appear."""
    check(len(edit["columns"]) == len(edit["weights"]), "edit columns and weights differ")
    check(len(set(edit["columns"])) == len(edit["columns"]), "duplicate intervention column")
    for column, weight in zip(edit["columns"], edit["weights"]):
        check(type(column) is int and 0 <= column < prefix_length and finite(weight) and weight >= 0, "invalid intervention edit")
        if weight:
            yield column, column + 1, prefix_length, weight


def memory_estimate(tokens, hidden_size, vocab_size, heads, kv_heads, layers, dtype_bytes):
    """Explicit conservative tensor accounting, excluding resident weights/vision workspace."""
    head_size = hidden_size // heads
    return {"sequence_tokens": tokens,
            "attention_and_mask_bytes": 4 * heads * tokens * tokens * 4,
            "full_sequence_output_logits_bytes": tokens * vocab_size * dtype_bytes,
            "two_cache_copies_bytes": 2 * layers * 2 * kv_heads * tokens * head_size * dtype_bytes,
            "qkv_and_hidden_workspace_bytes": 12 * tokens * hidden_size * 4}


class QwenPrefixBackend:
    """Pinned native-eager Qwen hooks; initialization accepts an already loaded model."""
    def __init__(self, evaluator, runtime, config):
        # Importing this module and running its stdlib tests never reaches here.
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv
        self.evaluator, self.model, self.torch = evaluator, evaluator.model, evaluator.torch
        self.processor, self.config, self.runtime = evaluator.processor, config, copy.deepcopy(runtime)
        self.rotary, self.repeat_kv = apply_multimodal_rotary_pos_emb, repeat_kv
        check(runtime["model_id"] == "Qwen/Qwen2-VL-2B-Instruct" and
              runtime["model_revision"] == "895c3a49bc3fa70a340399125c650a463535e71c", "unsupported model/revision")
        check(runtime["dtype"] in ("float32", "bfloat16") and runtime["attention_implementation"] == "eager", "requires validated FP32/BF16 eager path")
        check(importlib.metadata.version("transformers") == runtime["packages"]["transformers"] == "4.49.0", "unsupported Transformers implementation")
        check(self.model.config.model_type == "qwen2_vl" and not self.model.training, "requires eval-mode Qwen2-VL")
        check(not getattr(self.model.config, "use_sliding_window", False), "sliding-window attention is unsupported")
        check(not any("lora_" in name for name, _ in self.model.named_parameters()), "CAMA baseline requires frozen base weights")
        self.layers = self.model.model.layers
        check(len(self.layers) == 28 and max(config.stage2_layers) < len(self.layers), "unexpected decoder depth")
        check(all(type(layer.self_attn).__name__ == "Qwen2VLAttention" for layer in self.layers), "every decoder layer must use native eager attention")
        check(all(str(parameter.dtype) == "torch." + runtime["dtype"] for parameter in self.model.parameters()), "model parameter dtype differs from pinned runtime")
        self.absolute_tolerance = 0.0001 if runtime["device"] == "cpu" else 0.05
        self.source_identity = {"module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "native_attention_sha256": hashlib.sha256(Path(self.rotary.__code__.co_filename).read_bytes()).hexdigest(),
                                "runtime_sha256": digest(runtime), "version": VERSION}
        self.handles, self.parameter_flags = [], []
        self.clear()
        try:
            for parameter in self.model.parameters():
                self.parameter_flags.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
            for index in config.stage1_layers + config.stage2_layers:
                self.handles.append(self.layers[index].self_attn.register_forward_pre_hook(
                    self._attention_hook, with_kwargs=True))
            self.handles.append(self.layers[max(config.stage1_layers)].register_forward_hook(self._hidden_hook))
        except Exception:
            self.close()
            raise

    def clear(self):
        self.phase, self.plan, self.gains, self.relevance = "off", None, {}, None
        self.prompt_length, self.segments = 0, []
        self.seen = {"probe": [], "apply": [], "replay": []}
        self.cost = {"selector_probe_calls": 0, "modulated_prefill_calls": 0, "unmodified_prefill_calls": 0,
                     "candidate_or_greedy_cached_steps": 0, "reference_full_calls": 0,
                     "reference_stepwise_calls": 0, "reference_greedy_calls": 0,
                     "cache_branch_clones": 0, "trainable_parameters": 0, "optimizer_steps": 0,
                     "peak_memory_bytes": None, "peak_memory_status": "external measurement required; not inferred from an estimate"}

    def close(self):
        self.phase = "off"
        for handle in getattr(self, "handles", []):
            handle.remove()
        self.handles = []
        for parameter, required in getattr(self, "parameter_flags", []):
            parameter.requires_grad_(required)
        self.parameter_flags = []

    def _budget(self, tokens):
        check(tokens <= self.config.max_prefix_tokens + max(self.config.max_label_and_eos_tokens, self.config.max_generated_tokens),
              "total token limit exceeded")
        model = self.model.config
        estimate = memory_estimate(tokens, model.hidden_size, model.vocab_size, model.num_attention_heads,
                                   model.num_key_value_heads, len(self.layers), 4 if self.runtime["dtype"] == "float32" else 2)
        total = sum(value for key, value in estimate.items() if key.endswith("_bytes"))
        check(total <= self.config.max_transient_estimate_bytes, "estimated attention/logits/cache workspace exceeds declared limit")
        self.cost["maximum_transient_estimate_bytes"] = max(total, self.cost.get("maximum_transient_estimate_bytes", 0))
        self.cost["memory_estimate_components"] = estimate
        self.cost["memory_estimate_excludes"] = ["resident model weights", "vision encoder workspace", "allocator reserve"]

    def _qk(self, attention, hidden, kwargs):
        batch, length, _ = hidden.shape
        check(batch == 1, "CAMA supports batch size one")
        query = attention.q_proj(hidden).view(batch, length, -1, attention.head_dim).transpose(1, 2)
        key = attention.k_proj(hidden).view(batch, length, -1, attention.head_dim).transpose(1, 2)
        cos, sin = kwargs["position_embeddings"]
        query, key = self.rotary(query, key, cos, sin, attention.rope_scaling["mrope_section"])
        return query, self.repeat_kv(key, attention.num_key_value_groups)

    def _anchor_rows(self, attention, hidden, kwargs):
        query, key = self._qk(attention, hidden, kwargs)
        gains, raw = [], []
        for segment in self.segments:
            logits = self.torch.matmul(query[:, :, segment["anchors"], :],
                                      key[:, :, segment["image"], :].transpose(-2, -1)) / math.sqrt(attention.head_dim)
            rows = logits[0].float().mean(0).tolist()
            raw.append(rows)
            gains.append(single_gain(rows, self.config.epsilon))
        return gains, raw

    def _heads(self, attention, hidden, kwargs):
        query, key = self._qk(attention, hidden, kwargs)
        context = [position for segment in self.segments[:-1] for position in segment["image"] + segment["text"]]
        rows = self.segments[-1]["text"]
        logits = self.torch.matmul(query[:, :, rows, :], key[:, :, context, :].transpose(-2, -1)) / math.sqrt(attention.head_dim)
        scores = logits[0].float().sum(-1).mean(-1).tolist()
        return stable_top(scores, self.config.head_fraction), scores

    def _mask(self, attention, hidden, kwargs, edits):
        cells = [(edit, list(edit_positions(edit, self.prompt_length))) for edit in edits]
        if not any(rows for _, rows in cells):
            return kwargs  # exact zero-edit identity path, with no mask replacement
        length = hidden.shape[1]
        original = kwargs.get("attention_mask")
        check(original is not None and original.ndim == 4 and original.shape[0] == 1 and
              original.shape[-2] == length and original.shape[-1] >= length, "native eager causal mask unavailable")
        check(original.shape[1] == 1, "unexpected preexisting per-head intervention")
        # Every selected column is strictly before the receiving prefix row.
        # Clone the original mask: layer masks are shared by Transformers.
        per_head = any(edit["heads"] is not None and rows for edit, rows in cells)
        mask = original.expand(1, attention.num_heads if per_head else 1, length, original.shape[-1]).clone()
        for edit, rows in cells:
            heads = edit["heads"] if per_head and edit["heads"] is not None else list(range(mask.shape[1]))
            check(len(set(heads)) == len(heads) and all(type(head) is int and 0 <= head < mask.shape[1] for head in heads), "invalid selected head")
            for column, start, end, weight in rows:
                for head in heads:
                    block = mask[0, head, start:end, column]
                    check(bool((block >= 0).all().item()), "attempted edit of masked/padded attention positions")
                    block.add_(weight)
        return {**kwargs, "attention_mask": mask}

    def _attention_hook(self, attention, args, kwargs):
        if self.phase == "off":
            return None
        check(not args and "hidden_states" in kwargs, "unsupported attention calling convention")
        hidden = kwargs["hidden_states"]
        layer, length = attention.layer_idx, hidden.shape[1]
        check(hidden.shape[0] == 1 and length >= self.prompt_length, "intervention invoked on a cached/shortened input")
        cache = kwargs.get("past_key_value")
        check(cache is None or cache.get_seq_length(layer) == 0, "prefix/replay must start with an empty layer cache")
        if self.phase == "probe":
            check(length == self.prompt_length, "selector input contains appended candidate tokens")
            if layer in self.config.stage1_layers:
                self.gains[layer], raw = self._anchor_rows(attention, hidden, kwargs)
                self.plan["probe_anchor_mean_logits"][str(layer)] = raw
                self.seen["probe"].append(layer)
            return None
        if self.phase == "replay":
            self.seen["replay"].append(layer)
            return args, self._mask(attention, hidden, kwargs, self.plan["layer_edits"][str(layer)])
        check(self.phase == "apply" and length == self.prompt_length, "unknown or non-prefix intervention phase")
        edits, n = [], len(self.segments) - 1
        if layer in self.config.stage1_layers:
            for index, item in enumerate(self.plan["selected_images"]):
                factor = (n - index) / n if index < n else 0.0
                denominator = max(item["selected_scores"]) + self.config.epsilon
                weights = [self.config.stage1_scale * factor * value / denominator for value in item["selected_scores"]]
                edits.append({"heads": None, "columns": item["tokens"], "weights": weights})
        else:
            check(self.relevance is not None, "Stage II ran without modified Stage-I representations")
            heads, head_scores = self._heads(attention, hidden, kwargs)
            self.plan["selected_heads"][str(layer)] = {"heads": heads, "raw_logit_scores": head_scores}
            for index, segment in enumerate(self.segments[:-1]):
                columns = self.plan["selected_images"][index]["tokens"] + segment["text"]
                weight = self.config.stage2_scale * self.relevance[index] * (n - index) / n
                edits.append({"heads": heads, "columns": columns, "weights": [weight] * len(columns)})
        self.plan["layer_edits"][str(layer)] = edits
        self.seen["apply"].append(layer)
        return args, self._mask(attention, hidden, kwargs, edits)

    def _hidden_hook(self, module, args, output):
        if self.phase != "apply":
            return
        hidden = output[0]
        check(hidden.shape[1] == self.prompt_length, "Stage-I hidden-state selection received candidate tokens")
        vectors = []
        for segment, selected in zip(self.segments, self.plan["selected_images"]):
            vectors.append(self.torch.cat((hidden[0, selected["tokens"]].float().mean(0),
                                           hidden[0, segment["text"]].float().mean(0))))
        stacked = self.torch.stack(vectors)
        norms = stacked.norm(dim=-1, keepdim=True)
        check(bool(self.torch.isfinite(stacked).all().item()) and bool((norms > 0).all().item()), "nonfinite/zero joint representations")
        normalized = stacked / norms
        self.relevance = (normalized[:-1] @ normalized[-1]).softmax(0).tolist()
        check(all(finite(value) and value >= 0 for value in self.relevance) and abs(math.fsum(self.relevance) - 1) <= 1e-5,
              "invalid query-context relevance")
        self.plan["relevance"] = list(self.relevance)

    def _prefix_identity(self, prefix):
        return digest(prefix["metadata"])

    def prefill(self, prefix, enabled):
        self.clear()
        inputs = prefix["inputs"]
        ids = inputs["input_ids"][0].tolist()
        check(inputs["input_ids"].shape[0] == 1 and len(ids) <= self.config.max_prefix_tokens, "prefix token/batch limit exceeded")
        check("pixel_values_videos" not in inputs and "video_grid_thw" not in inputs, "video inputs unsupported")
        check(inputs["attention_mask"].ndim == 2 and bool((inputs["attention_mask"] == 1).all().item()), "padding unsupported")
        self.prompt_length = len(ids)
        self._budget(self.prompt_length)
        self.segments = classification_segments(ids, self.processor.tokenizer, self.model.config.image_token_id)
        check([len(segment["image"]) for segment in self.segments] == prefix["metadata"]["per_image_visual_token_counts"], "CAMA image segmentation differs from processor grids")
        active = enabled and len(self.segments) > 1
        self.plan = {"version": VERSION, "source_identity": self.source_identity, "config": asdict(self.config),
                     "enabled": enabled, "active": active, "zero_shot_identity": len(self.segments) == 1,
                     "prefix_identity": self._prefix_identity(prefix), "prefix_token_ids": ids,
                     "segments": copy.deepcopy(self.segments), "selected_images": [], "selected_heads": {},
                     "layer_edits": {}, "probe_anchor_mean_logits": {}, "relevance": [],
                     "selector_math": "native-dtype QK; float32 head average; Python float64 softmax/gain; stable ceiling ties",
                     "disclosure": DISCLOSURE}
        try:
            with self.torch.inference_mode():
                if active:
                    self.phase = "probe"
                    probe = self.model(**inputs, use_cache=False, return_dict=True)
                    del probe
                    self.cost["selector_probe_calls"] += 1
                    check(self.seen["probe"] == list(self.config.stage1_layers), "missing or repeated Stage-I probe")
                    for index, segment in enumerate(self.segments):
                        values = [math.fsum(self.gains[layer][index][token] for layer in self.config.stage1_layers)
                                  for token in range(len(segment["image"]))]
                        selected = stable_top(values, self.config.token_fraction)
                        self.plan["selected_images"].append({"tokens": [segment["image"][i] for i in selected],
                                                            "selected_scores": [values[i] for i in selected], "all_scores": values})
                    self.phase = "apply"
                result = self.model(**inputs, use_cache=True, return_dict=True)
                self.cost["modulated_prefill_calls" if active else "unmodified_prefill_calls"] += 1
                if active:
                    expected = list(self.config.stage1_layers + self.config.stage2_layers)
                    check(self.seen["apply"] == expected and set(self.plan["layer_edits"]) == {str(value) for value in expected},
                          "both CAMA stages must execute exactly once")
                first = result.logits[0, -1].float().log_softmax(-1)
                state = {"first": first, "cache": result.past_key_values, "rope": self.model.rope_deltas.clone(),
                         "position": self.prompt_length, "plan": copy.deepcopy(self.plan)}
                check(state["cache"].get_seq_length() == self.prompt_length, "prefill cache length differs")
                del result
                return state
        finally:
            self.phase = "off"

    def clone(self, state):
        check(state["cache"].get_seq_length() == state["position"], "a previous branch mutated shared prefix cache")
        self.cost["cache_branch_clones"] += 1
        self.model.rope_deltas = state["rope"].clone()
        return {"cache": copy.deepcopy(state["cache"]), "position": state["position"]}

    def advance(self, branch, token):
        check(self.phase == "off", "candidate continuation attempted with active selector/intervention")
        check(branch["cache"].get_seq_length() == branch["position"], "cache branch position changed")
        with self.torch.inference_mode():
            result, branch["cache"] = self.evaluator._next(token, branch["cache"], branch["position"])
        branch["position"] += 1
        self.cost["candidate_or_greedy_cached_steps"] += 1
        return result

    @staticmethod
    def release(state):
        if "plan" in state and "cache" in state:
            check(state["cache"].get_seq_length() == state["position"], "shared prefix cache changed during branch scoring")
        state.clear()

    @staticmethod
    def value(scores, token):
        check(0 <= token < len(scores), "token outside model vocabulary")
        value = float(scores[token].item())
        check(finite(value) and value <= 1e-6, "invalid token log probability")
        return value

    @staticmethod
    def argmax(scores):
        return int(scores.argmax().item())

    def _replay(self, prefix, appended, plan):
        check(plan["prefix_identity"] == self._prefix_identity(prefix), "reference prefix differs from selected intervention")
        check(plan["source_identity"] == self.source_identity and digest(plan["config"]) == digest(asdict(self.config)), "reference intervention configuration changed")
        n = len(plan["prefix_token_ids"])
        check(prefix["inputs"]["input_ids"][0].tolist() == plan["prefix_token_ids"], "reference prefix tokens changed")
        check(len(appended) <= max(self.config.max_label_and_eos_tokens, self.config.max_generated_tokens), "reference continuation exceeds budget")
        self._budget(n + len(appended))
        inputs = dict(prefix["inputs"])
        continuation = self.torch.tensor([appended], dtype=self.torch.long, device=self.evaluator.device)
        inputs["input_ids"] = self.torch.cat((inputs["input_ids"], continuation), dim=1)
        inputs["attention_mask"] = self.torch.ones_like(inputs["input_ids"])
        self.plan, self.prompt_length = copy.deepcopy(plan), n
        self.seen["replay"] = []
        self.phase = "replay" if plan["active"] else "off"
        try:
            result = self.model(**inputs, use_cache=False, return_dict=True)
            if plan["active"]:
                check(self.seen["replay"] == list(self.config.stage1_layers + self.config.stage2_layers), "reference failed to replay both stages")
            return result
        finally:
            self.phase = "off"

    def reference(self, prefix, candidate, plan, mode):
        check(mode in ("full", "stepwise"), "unknown reference mode")
        n, values = len(plan["prefix_token_ids"]), []
        with self.torch.inference_mode():
            if mode == "full":
                result = self._replay(prefix, candidate, plan)
                selected = result.logits[0, n - 1:n + len(candidate) - 1].float().log_softmax(-1)
                values = [self.value(row, token) for row, token in zip(selected, candidate)]
                del result, selected
                self.cost["reference_full_calls"] += 1
            else:
                for index, token in enumerate(candidate):
                    result = self._replay(prefix, candidate[:index], plan)
                    values.append(self.value(result.logits[0, -1].float().log_softmax(-1), token))
                    del result
                    self.cost["reference_stepwise_calls"] += 1
        return values

    def reference_greedy(self, prefix, plan, stop_ids, maximum):
        generated = []
        with self.torch.inference_mode():
            for _ in range(maximum):
                result = self._replay(prefix, generated, plan)
                token = int(result.logits[0, -1].argmax().item())
                generated.append(token)
                del result
                self.cost["reference_greedy_calls"] += 1
                if token in stop_ids:
                    break
        return generated


class CamaAdapter:
    """Own temporary hooks on one root-loaded evaluator; close before reusing it."""
    def __init__(self, evaluator, runtime, config=None):
        self.evaluator, self.config = evaluator, config or CamaConfig()
        self.backend = QwenPrefixBackend(evaluator, runtime, self.config)
        self.scorer = SharedPrefixScorer(self.backend, self.config)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.backend.close()

    def score_encoded(self, inputs, metadata, enabled=True, verify=False):
        # Metadata passed to selectors contains prefix identity only, not any
        # candidate spans, labels, episode classes, or answer token contracts.
        prefix = {"inputs": inputs, "metadata": {key: copy.deepcopy(metadata[key]) for key in
                  ("raw_rendered_prompt", "input_token_ids", "image_grid_thw", "per_image_visual_token_counts")}}
        old_rope = self.evaluator.model.rope_deltas
        old_rope = None if old_rope is None else old_rope.clone()
        try:
            result = self.scorer.score(prefix, metadata["candidate_tokens"], self.evaluator.eos,
                                       self.evaluator.stop_ids, self.evaluator.max_new_tokens, enabled, verify)
            result["cost"] = copy.deepcopy(self.backend.cost)
            result["cost"].update(candidate_continuation_tokens=sum(len(row["scored_token_ids"]) - 1 for row in metadata["candidate_tokens"]),
                                  greedy_continuation_calls=max(0, len(result["greedy_token_ids"]) - 1),
                                  prefix_tokens=len(metadata["input_token_ids"]),
                                  visual_tokens=sum(metadata["per_image_visual_token_counts"]))
            return result
        finally:
            self.backend.clear()
            self.evaluator.model.rope_deltas = old_rope

    def evaluate(self, episode, image_loader, natural_labels, enabled=True, verify=False):
        from .protocol import parse_diagnostics
        self.evaluator.synchronize()
        started = time.monotonic()
        messages, images = classification_messages(episode, image_loader)
        inputs, metadata = self.evaluator.encode(messages, images, episode["label_space"])
        result = self.score_encoded(inputs, metadata, enabled, verify)
        answer = self.evaluator.processor.tokenizer.decode(result["greedy_token_ids"], skip_special_tokens=True,
                                                          clean_up_tokenization_spaces=False)
        parsing = parse_diagnostics(answer, episode["label_space"], natural_labels=natural_labels)
        self.evaluator.synchronize()
        return {"status": "ok", "episode": episode, **metadata, **result, "greedy_answer": answer, "parser": parsing,
                "method": "CAMA independent classification adaptation" if enabled else "CAMA matched frozen prompt/backend control",
                "historical_strict_disagreement": parsing["legacy_prediction"] != parsing["prediction"],
                "sum_mean_disagreement": result["scores"]["sum_prediction"] != result["scores"]["mean_prediction"],
                "generation_reached_limit": len(result["greedy_token_ids"]) == self.evaluator.max_new_tokens and
                    result["greedy_token_ids"][-1] not in self.evaluator.stop_ids,
                "seconds": time.monotonic() - started, "model_integration_verified": verify,
                "scientific_gate_approved": False}
