"""Accepted answer-mask, loss-input and parameter-identity operations."""
from __future__ import annotations
import hashlib, json, math
from .qwen_base import render_messages


def complete_answer_mask(prefix_ids, label_ids, eos_token_id):
    """Return exact IDs, training labels, and causally shifted prediction indices."""
    prefix_ids, label_ids = list(prefix_ids), list(label_ids)
    if not prefix_ids or not label_ids:
        raise ValueError("nonempty prompt and complete label are mandatory")
    if type(eos_token_id) is not int or eos_token_id < 0:
        raise ValueError("EOS must be a valid integer token ID")
    if any(type(token) is not int or token < 0 for token in prefix_ids + label_ids):
        raise ValueError("token IDs must be nonnegative integers")
    if eos_token_id in label_ids:
        raise ValueError("label already contains EOS")
    answer = label_ids + [eos_token_id]
    return {"input_ids": prefix_ids + answer, "labels": [-100] * len(prefix_ids) + answer,
            "answer_token_ids": answer, "prefix_length": len(prefix_ids),
            "answer_positions": list(range(len(prefix_ids), len(prefix_ids) + len(answer))),
            "prediction_positions": list(range(len(prefix_ids) - 1, len(prefix_ids) + len(answer) - 1))}



def training_batch(evaluator, episode, image_loader):
    messages, images = render_messages(episode, image_loader)
    inputs, metadata = evaluator.encode(messages, images, episode["label_space"])
    candidate = metadata["candidate_tokens"][episode["query"]["class_id"]]
    if candidate["label"] != episode["supervised_label"]:
        raise ValueError("training target conflicts with the demonstration-label objective")
    mask = complete_answer_mask(inputs["input_ids"][0].tolist(), candidate["label_token_ids"], evaluator.eos)
    torch = evaluator.torch
    batch = dict(inputs)
    batch["input_ids"] = torch.tensor([mask["input_ids"]], dtype=torch.long, device=evaluator.device)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch["labels"] = torch.tensor([mask["labels"]], dtype=torch.long, device=evaluator.device)
    return batch, {"episode": episode, "encoding": metadata, "supervision": mask}



def parameter_hashes(model, torch, adapter_only=False):
    """Exact parameter bytes, copied one tensor at a time; never save full weights."""
    result = {}
    for name, parameter in model.named_parameters():
        is_adapter = ".lora_A." in name or ".lora_B." in name
        if adapter_only and not is_adapter:
            continue
        header = json.dumps({"shape": list(parameter.shape), "dtype": str(parameter.dtype)}, sort_keys=True).encode()
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(header + b"\0")
        digest.update(raw)
        result[name] = digest.hexdigest()
    if not result:
        raise ValueError("no parameter tensors available for checkpoint verification")
    return result



def compare_roundtrip(before, after, tolerance):
    for key in ("episode", "input_token_ids", "candidate_tokens", "greedy_token_ids"):
        if before[key] != after[key]:
            raise ValueError(f"checkpoint reload changed {key}")
    if before["scores"]["sum_prediction"] != after["scores"]["sum_prediction"]:
        raise ValueError("checkpoint reload changed primary prediction")
    left, right = before["per_token_label_log_probabilities"], after["per_token_label_log_probabilities"]
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("checkpoint reload changed scored token spans")
    errors = [abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b)]
    if not errors or not all(math.isfinite(error) for error in errors) or max(errors) > tolerance:
        raise ValueError("checkpoint reload changed label scores beyond declared tolerance")
    return max(errors)



def require_positive_norms(values, description):
    if (not values or not all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values)
            or not any(value > 0 for value in values)):
        raise ValueError(f"{description} must include a positive finite norm; decay-only change is insufficient")



def require_error_bound(error, tolerance, expected, description):
    if (type(error) not in (int, float) or type(tolerance) not in (int, float)
            or not math.isfinite(error) or not math.isfinite(tolerance)
            or error < 0 or tolerance != expected or error > tolerance):
        raise ValueError(f"{description} requires a finite nonnegative error within the declared tolerance")
