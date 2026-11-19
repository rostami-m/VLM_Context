"""Accepted Qwen native scoring paths, with independent teacher-forcing checks."""
from __future__ import annotations
import copy, importlib.metadata, json, math, os, random, sys, time
from pathlib import Path
from .protocol import config_hash, parse_diagnostics, sequence_likelihoods, summarize_predictions
from .io import sha256 as sha256_file, write_json as immutable_json


MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"



MODEL_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"



SAMPLERS = ["legacy_class_cycle_with_replacement", "randomized_balanced_without_replacement"]



REMAP = ["dax", "wug", "blicket", "koba"]



def render_messages(episode, image_loader):
    content = []
    images = []
    for support in episode["supports"]:
        image = image_loader(support["image_id"])
        images.append(image)
        content.extend([{"type": "image"},
                        {"type": "text", "text": "This image shows: " +
                         episode["label_space"][support["class_id"]] + "."}])
    images.append(image_loader(episode["query"]["image_id"]))
    content.extend([{"type": "image"}, {"type": "text", "text":
                    "Classify the last image. Answer with exactly one of [" +
                    ", ".join(episode["displayed_options"]) + "] and nothing else."}])
    return [{"role": "user", "content": content}], images



def validate_answer_boundary(tokenizer, prompt, labels, eos_token_id):
    """Fail on label-dependent prefix retokenization; no longest-prefix repair."""
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prefix:
        raise ValueError("empty prompt token sequence")
    candidates = []
    for label in labels:
        full = tokenizer(prompt + label, add_special_tokens=False)["input_ids"]
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"candidate changes shared prompt tokenization: {label!r}")
        suffix = full[len(prefix):]
        if tokenizer.decode(suffix, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != label:
            raise ValueError("candidate token span does not decode to exactly the label")
        if eos_token_id in suffix:
            raise ValueError("answer contains EOS")
        candidates.append({"label": label, "label_token_ids": suffix,
                           "scored_token_ids": suffix + [eos_token_id]})
    return {"prefix_token_ids_before_visual_expansion": prefix,
            "candidate_tokens": candidates,
            "answer_boundary": "exact chat generation prefix, no added leading whitespace",
            "eos_convention": "append and score exactly one <|im_end|> token",
            "eos_token_id": eos_token_id}



class QwenEvaluator:
    def __init__(self, runtime, seed=0, max_new_tokens=8):
        import numpy as np
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.torch = torch
        self.device = runtime["device"]
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested MPS is unavailable; refusing CPU fallback")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA is unavailable")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        # The pinned eager Qwen implementation replaces +/-inf attention scores
        # with zero in float16. Adding its finite causal-mask minimum can overflow
        # to -inf and expose future tokens. The retained smoke diagnostic fails
        # cache/full-prefix agreement in fp16; bf16 preserves the mask range.
        if runtime["dtype"] not in ("float32", "bfloat16"):
            raise ValueError("this validated evaluator requires float32 or bfloat16")
        dtype = getattr(torch, runtime["dtype"])
        self.processor = AutoProcessor.from_pretrained(runtime["snapshot"],
                                                       local_files_only=True,
                                                       trust_remote_code=False, use_fast=False)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            runtime["snapshot"], torch_dtype=dtype, local_files_only=True,
            trust_remote_code=False, attn_implementation="eager").to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.eos = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.processor.tokenizer.decode([self.eos], skip_special_tokens=False) != "<|im_end|>":
            raise ValueError("expected Qwen end-of-turn token is unavailable")
        stop = self.model.generation_config.eos_token_id
        self.stop_ids = set(stop if isinstance(stop, list) else [stop]) | {self.eos}
        self.identity = {"parameter_dtype_counts": self._dtype_counts(),
                         "image_processor": self.processor.image_processor.to_dict(),
                         "tokenizer_class": type(self.processor.tokenizer).__name__,
                         "end_of_turn_token_id": self.eos, "generation_stop_ids": sorted(self.stop_ids),
                         "max_new_tokens": max_new_tokens, "do_sample": False,
                         "seed_python_numpy_torch": seed, "deterministic_algorithms": True,
                         "input_image_size": [112, 112], "image_conversion": "RGB"}

    def _dtype_counts(self):
        counts = {}
        for parameter in self.model.parameters():
            name = str(parameter.dtype)
            counts[name] = counts.get(name, 0) + parameter.numel()
        return counts

    def synchronize(self):
        if self.device == "mps":
            self.torch.mps.synchronize()
        elif self.device == "cuda":
            self.torch.cuda.synchronize()

    def encode(self, messages, images, labels):
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        boundary = validate_answer_boundary(self.processor.tokenizer, prompt, labels, self.eos)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.device)
        ids = inputs["input_ids"][0].tolist()
        # Visual expansion changes only image-pad runs. Validate against the
        # tokenizer's pre-expansion prompt instead of assuming identical length.
        image_token = self.model.config.image_token_id
        collapse = lambda seq: [v for i, v in enumerate(seq)
                                if v != image_token or i == 0 or seq[i - 1] != image_token]
        if collapse(ids) != collapse(boundary["prefix_token_ids_before_visual_expansion"]):
            raise ValueError("processor altered prompt beyond visual-token expansion")
        grids = inputs["image_grid_thw"].tolist()
        merge = self.processor.image_processor.merge_size
        visual_tokens = [math.prod(grid) // merge ** 2 for grid in grids]
        if sum(visual_tokens) != ids.count(image_token) or len(grids) != len(images):
            raise ValueError("per-image visual token accounting mismatch")
        metadata = {"raw_rendered_prompt": prompt, "input_token_ids": ids,
                    "input_token_count": len(ids), "image_grid_thw": grids,
                    "per_image_visual_token_counts": visual_tokens,
                    "input_tensor_dtypes": {k: str(v.dtype) for k, v in inputs.items()},
                    "input_tensor_shapes": {k: list(v.shape) for k, v in inputs.items()},
                    **boundary}
        return inputs, metadata

    def _next(self, token, cache, position):
        torch = self.torch
        output = self.model(input_ids=torch.tensor([[token]], device=self.device),
                            attention_mask=torch.ones((1, position + 1), dtype=torch.long,
                                                      device=self.device),
                            past_key_values=cache, cache_position=torch.tensor([position], device=self.device),
                            use_cache=True, return_dict=True)
        return output.logits[0, -1].float().log_softmax(-1), output.past_key_values

    def _cached_candidates(self, inputs, metadata):
        torch = self.torch
        n = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            prefix = self.model(**inputs, use_cache=True, return_dict=True)
            first = prefix.logits[0, -1].float().log_softmax(-1)
            base_cache = prefix.past_key_values
            rope = self.model.rope_deltas.clone()
            del prefix
            logps = []
            for candidate in metadata["candidate_tokens"]:
                self.model.rope_deltas = rope.clone()
                cache, next_logp = copy.deepcopy(base_cache), first
                values = []
                ids = candidate["scored_token_ids"]
                for index, token in enumerate(ids):
                    values.append(float(next_logp[token].item()))
                    if index + 1 < len(ids):
                        next_logp, cache = self._next(token, cache, n + index)
                logps.append(values)
                del cache
            self.model.rope_deltas = rope.clone()
            cache, next_logp, generated, generated_logps = copy.deepcopy(base_cache), first, [], []
            for index in range(self.max_new_tokens):
                token = int(next_logp.argmax().item())
                generated.append(token)
                generated_logps.append(float(next_logp[token].item()))
                if token in self.stop_ids:
                    break
                next_logp, cache = self._next(token, cache, n + index)
            del cache, base_cache
        return logps, generated, generated_logps

    def independent_full_forward(self, inputs, candidate_ids):
        """Independent full-input teacher forcing, including the causal shift."""
        torch = self.torch
        n = inputs["input_ids"].shape[1]
        full = dict(inputs)
        full["input_ids"] = torch.cat((inputs["input_ids"],
                                       torch.tensor([candidate_ids], device=self.device)), dim=1)
        full["attention_mask"] = torch.ones_like(full["input_ids"])
        with torch.inference_mode():
            out = self.model(**full, use_cache=False, return_dict=True)
            selected = out.logits[0, n - 1:n + len(candidate_ids) - 1].float().log_softmax(-1)
            values = [float(selected[i, token].item()) for i, token in enumerate(candidate_ids)]
        return values

    def independent_stepwise(self, inputs, candidate_ids):
        """No-cache prefix recomputation for each next-token probability."""
        torch = self.torch
        values = []
        with torch.inference_mode():
            for index, token in enumerate(candidate_ids):
                step = dict(inputs)
                previous = torch.tensor([candidate_ids[:index]], dtype=torch.long, device=self.device)
                step["input_ids"] = torch.cat((inputs["input_ids"], previous), dim=1)
                step["attention_mask"] = torch.ones_like(step["input_ids"])
                out = self.model(**step, use_cache=False, return_dict=True)
                values.append(float(out.logits[0, -1].float().log_softmax(-1)[token].item()))
        return values

    def evaluate(self, episode, image_loader, natural_labels, verify=False):
        self.synchronize()
        started = time.monotonic()
        messages, images = render_messages(episode, image_loader)
        inputs, metadata = self.encode(messages, images, episode["label_space"])
        logps, generated, generated_logps = self._cached_candidates(inputs, metadata)
        scores = sequence_likelihoods(logps)
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        parsing = parse_diagnostics(raw, episode["label_space"], natural_labels=natural_labels)
        validation = None
        if verify:
            fulls, steps = [], []
            for candidate in metadata["candidate_tokens"]:
                fulls.append(self.independent_full_forward(inputs, candidate["scored_token_ids"]))
                steps.append(self.independent_stepwise(inputs, candidate["scored_token_ids"]))
            errors = [abs(a - b) for cached, full, step in zip(logps, fulls, steps)
                      for a, b in list(zip(cached, full)) + list(zip(cached, step))]
            tolerance = 0.05 if self.device != "cpu" else 0.0001
            if max(errors) > tolerance:
                raise ValueError(f"cached/full/stepwise log probability mismatch: {max(errors)} > {tolerance}")
            full_scores = sequence_likelihoods(fulls)
            if scores["sum_prediction"] != full_scores["sum_prediction"]:
                raise ValueError("cache/full-forward primary argmax disagrees")
            with self.torch.inference_mode():
                hf_out = self.model.generate(**inputs, do_sample=False,
                                             max_new_tokens=self.max_new_tokens)
            hf_tokens = hf_out[0, inputs["input_ids"].shape[1]:].tolist()
            if hf_tokens != generated:
                raise ValueError("cached greedy generation differs from model.generate")
            validation = {"full_forward_token_logps": fulls,
                          "independent_stepwise_token_logps": steps,
                          "maximum_absolute_logp_difference": max(errors),
                          "absolute_tolerance": tolerance,
                          "generation_matches_transformers_generate": True}
        self.synchronize()
        ties = {name: [i for i, score in enumerate(scores[name]) if score == max(scores[name])]
                for name in ("sum", "mean")}
        return {"status": "ok", "episode": episode, **metadata,
                "per_token_label_log_probabilities": logps, "scores": scores, "ties": ties,
                "greedy_token_ids": generated, "greedy_token_log_probabilities": generated_logps,
                "greedy_answer": raw, "generation_reached_limit":
                len(generated) == self.max_new_tokens and generated[-1] not in self.stop_ids,
                "parser": parsing, "historical_strict_disagreement":
                parsing["legacy_prediction"] != parsing["prediction"],
                "sum_mean_disagreement": scores["sum_prediction"] != scores["mean_prediction"],
                "independent_validation": validation, "seconds": time.monotonic() - started}
