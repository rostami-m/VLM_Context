"""Accepted common-prompt native, CAMA and calibration operations."""
from __future__ import annotations
import copy, hashlib, json, math, time
from .io import require as check
from .protocol import config_hash
from .calibration import SCORE_DOMAIN, batch_calibration, contextual_calibration
from .cama import CamaAdapter, CamaConfig, classification_messages, likelihoods


NEUTRAL = {"mode": "RGB", "size": [112, 112], "color": [127, 127, 127]}



def validate_score_record(record, episode):
    check(record["status"] == "ok" and record["episode"] == episode, "record/episode identity differs")
    labels, eos = episode["label_space"], record["eos_token_id"]
    candidates, rows = record["candidate_tokens"], record["per_token_label_log_probabilities"]
    check(type(eos) is int and eos >= 0 and len(candidates) == len(rows) == 4, "valid EOS and four candidate traces required")
    for label, candidate, values in zip(labels, candidates, rows):
        tokens = candidate["label_token_ids"]
        check(candidate["label"] == label and bool(tokens) and all(type(t) is int and t >= 0 for t in tokens) and eos not in tokens and
              candidate["scored_token_ids"] == tokens + [eos] and len(values) == len(tokens) + 1,
              "answer trace must include all label tokens plus one EOS")
    scores = likelihoods(rows)
    check(record["scores"] == scores, "saved scores disagree with raw token log probabilities")
    return scores["sum"]



def neutral_episode(episode):
    result = copy.deepcopy(episode)
    header = json.dumps(["RGB", 112, 112], separators=(",", ":")).encode()
    pixel_hash = hashlib.sha256(header + b"\0" + bytes([127]) * (112 * 112 * 3)).hexdigest()
    result["query"] = {"image_id": "baseline-neutral-rgb127-112", "content_sha256": pixel_hash}
    result["query_role"] = "content_free_not_an_evaluation_query"
    return result



def calibrate_panel(panel, frozen_records, neutral_record=None):
    """Only scores/context/label strings reach calibration; truth is read later."""
    episodes = panel["episodes"]
    check(len(frozen_records) == len(episodes) == 80, "incomplete shared deployment panel")
    rows = [validate_score_record(record, episode) for record, episode in zip(frozen_records, episodes)]
    first = episodes[0]
    check(first["condition"] == "standard", "calibration is registered for standard labels only")
    contexts = [episode["context_sha256"] for episode in episodes]
    labels = first["label_space"]
    check(all(episode["label_space"] == labels and
              {k: v for k, v in episode.items() if k not in ("query", "query_role")} ==
              {k: v for k, v in first.items() if k not in ("query", "query_role")} for episode in episodes),
          "panel pooled different supports/mappings/options")
    check(neutral_record is not None, "one gray query score is required per context")
    neutral = validate_score_record(neutral_record, neutral_episode(first))
    cc = [contextual_calibration(row, [neutral], label_order=labels, content_free_label_order=labels,
          query_context=context, content_free_contexts=[first["context_sha256"]], score_domain=SCORE_DOMAIN)
          for row, context in zip(rows, contexts)]
    bc = batch_calibration(rows, label_order=labels, context_hashes=contexts, score_domain=SCORE_DOMAIN,
                           batch_role="unlabeled_deployment")
    return {"panel_id": panel["panel_id"], "cc": cc, "bc": bc,
            "uses_ground_truth_labels": False, "decision_interface": "restricted candidate likelihood classification"}



def compare_scores(left, right, *, tolerance=0.05, reverse_candidates=False):
    left_rows = left["per_token_label_log_probabilities"]
    right_rows = right["per_token_label_log_probabilities"]
    if reverse_candidates:
        right_rows = list(reversed(right_rows))
    check(len(left_rows) == len(right_rows) == 4 and all(len(a) == len(b) for a, b in zip(left_rows, right_rows)),
          "parity trace lengths differ")
    errors = [abs(a - b) for row, other in zip(left_rows, right_rows) for a, b in zip(row, other)]
    check(all(math.isfinite(v) for v in errors) and max(errors) <= tolerance, "parity score tolerance failed")
    check(likelihoods(left_rows)["sum_prediction"] == likelihoods(right_rows)["sum_prediction"], "parity argmax differs")
    check(left["greedy_token_ids"] == right["greedy_token_ids"], "parity greedy tokens differ")
    return {"maximum_absolute_logp_difference": max(errors), "tolerance": tolerance,
            "sum_argmax_equal": True, "greedy_tokens_equal": True}



def native_control(evaluator, inputs, metadata, verify=True):
    """Unwrapped scorer on the identical CAMA classification prompt."""
    rows, generated, generated_logps = evaluator._cached_candidates(inputs, metadata)
    result = {"per_token_label_log_probabilities": rows, "scores": likelihoods(rows),
              "greedy_token_ids": generated, "greedy_token_log_probabilities": generated_logps}
    if verify:
        full = [evaluator.independent_full_forward(inputs, c["scored_token_ids"]) for c in metadata["candidate_tokens"]]
        steps = [evaluator.independent_stepwise(inputs, c["scored_token_ids"]) for c in metadata["candidate_tokens"]]
        tolerance = 0.0001 if evaluator.device == "cpu" else 0.05
        for reference in (full, steps):
            compare_scores(result, {"per_token_label_log_probabilities": reference, "greedy_token_ids": generated}, tolerance=tolerance)
        with evaluator.torch.inference_mode():
            tokens = evaluator.model.generate(**inputs, do_sample=False, max_new_tokens=evaluator.max_new_tokens)
        native_tokens = tokens[0, inputs["input_ids"].shape[1]:].tolist()
        check(native_tokens == generated, "unwrapped greedy differs from native generate")
        result["independent_validation"] = {"full_forward_token_logps": full, "stepwise_token_logps": steps,
                                            "native_greedy_token_ids": native_tokens}
    return result



def evaluate_baseline(evaluator, runtime, episode, loader, method, verify=False):
    """Same-prompt inference primitive; caller owns authorization, lock and writes."""
    check(method in ("frozen", "cama"), "calibration is score analysis, not a separate model")
    evaluator.synchronize()
    started = time.monotonic()
    messages, images = classification_messages(episode, loader)
    inputs, metadata = evaluator.encode(messages, images, episode["label_space"])
    if method == "cama":
        with CamaAdapter(evaluator, runtime) as adapter:
            result = adapter.score_encoded(inputs, metadata, enabled=True, verify=verify)
    else:
        result = native_control(evaluator, inputs, metadata, verify=verify)
        result["cost"] = {"unmodified_prefill_calls": 1, "selector_probe_calls": 0,
            "candidate_continuation_tokens": sum(len(c["scored_token_ids"]) - 1 for c in metadata["candidate_tokens"]),
            # The frozen E1 scorer also advances after its last generated token
            # when the generation limit is reached without a stop token.
            "greedy_continuation_calls": len(result["greedy_token_ids"]) -
                int(result["greedy_token_ids"][-1] in evaluator.stop_ids),
            "prefix_tokens": len(metadata["input_token_ids"]),
            "visual_tokens": sum(metadata["per_image_visual_token_counts"]),
            "verification_forwards_included": verify, "trainable_parameters": 0, "optimizer_steps": 0,
            "reference_full_calls": 4 if verify else 0,
            "reference_stepwise_calls": sum(len(c["scored_token_ids"]) for c in metadata["candidate_tokens"]) if verify else 0,
            "native_generate_reference_calls": int(verify),
            "peak_memory_bytes": None, "peak_memory_status": "external measurement required"}
    evaluator.synchronize()
    return {"status": "ok", "episode": copy.deepcopy(episode), **metadata, **result,
            "baseline_method": method, "runtime_sha256": config_hash(runtime),
            "seconds": time.monotonic() - started, "scientific_gate_approved": False}



def neutral_loader(original_loader):
    """Lazily create only the declared gray image when actual inference requests it."""
    def load(image_id):
        if image_id != "baseline-neutral-rgb127-112":
            return original_loader(image_id)
        from PIL import Image
        return Image.new("RGB", tuple(NEUTRAL["size"]), tuple(NEUTRAL["color"]))
    return load
