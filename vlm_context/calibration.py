"""CPU score-space calibration primitives.

CC: Zhao et al. official run_classification.py, diagonal_W and
get_p_content_free: average unnormalized probabilities, then normalize the prior.
https://github.com/tonyzhaozh/few-shot-learning/blob/main/run_classification.py
BC: Zhou et al., arXiv:2309.17249v3 equations 1-2 and Appendix D: subtract
column means in the declared log-score domain. No softmax/probability-mean swap.
https://arxiv.org/html/2309.17249v3

Complete-label-plus-EOS inputs and gray-image content-free prompts are explicit
multimodal adaptations, not reproductions of original single-token text runs.
Context hashes must cover supports/order, mapping, option order, prompt, model,
checkpoint and scorer. Query content is excluded. No labels enter calibration.
"""
from __future__ import annotations

import math
import re

SCORE_DOMAIN = "sum_complete_label_plus_eos_log_probability"


def _row(values, n):
    row = list(values)
    if len(row) != n or not all(type(v) in (int, float) and math.isfinite(v) and v <= 0 for v in row):
        raise ValueError("expected finite nonpositive complete-sequence log probabilities")
    return row


def _labels(values):
    values = list(values)
    if len(values) < 2 or not all(isinstance(v, str) and v.strip() for v in values) or len(set(values)) != len(values):
        raise ValueError("at least two distinct explicit label strings are required")
    return values


def _context(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("explicit SHA256 context identity required")
    return value


def _logsumexp(values):
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def _prediction(scores):
    top = max(scores)
    ties = [i for i, score in enumerate(scores) if score == top]
    return {"prediction": ties[0], "ties": ties}


def contextual_calibration(query_log_scores, content_free_log_scores, *, label_order,
                           content_free_label_order, query_context, content_free_contexts,
                           score_domain):
    if score_domain != SCORE_DOMAIN:
        raise ValueError("calibration requires the declared complete-sequence log-score domain")
    labels = _labels(label_order)
    if list(content_free_label_order) != labels:
        raise ValueError("content-free label order differs from query label order")
    query = _row(query_log_scores, len(labels))
    rows = [_row(values, len(labels)) for values in content_free_log_scores]
    contexts = list(content_free_contexts)
    if not rows or len(contexts) != len(rows) or any(_context(v) != _context(query_context) for v in contexts):
        raise ValueError("content-free prompts must preserve the exact query context")
    # Arithmetic mean in probability space, evaluated stably in log space.
    mean_log_prob = [_logsumexp([row[c] for row in rows]) - math.log(len(rows)) for c in range(len(labels))]
    normalizer = _logsumexp(mean_log_prob)
    prior = [value - normalizer for value in mean_log_prob]
    scores = [value - bias for value, bias in zip(query, prior)]
    return {"method": "contextual_calibration_diagonal_ratio", "score_domain": score_domain,
            "log_prior": prior, "calibrated_log_scores": scores, "content_free_count": len(rows),
            "context_sha256": query_context, "label_order": labels, **_prediction(scores)}


def batch_calibration(log_scores, *, label_order, context_hashes, score_domain, batch_role):
    if score_domain != SCORE_DOMAIN:
        raise ValueError("batch calibration requires complete-sequence log scores")
    if batch_role not in ("unlabeled_development", "unlabeled_deployment"):
        raise ValueError("declare the unlabeled batch role")
    labels = _labels(label_order)
    rows = [_row(values, len(labels)) for values in log_scores]
    contexts = [_context(value) for value in context_hashes]
    if len(rows) < 2 or len(contexts) != len(rows) or len(set(contexts)) != 1:
        raise ValueError("BC requires at least two queries with the same frozen context")
    bias = [math.fsum(row[c] for row in rows) / len(rows) for c in range(len(labels))]
    calibrated = [[value - bias[c] for c, value in enumerate(row)] for row in rows]
    return {"method": "batch_calibration_log_score_centering", "score_domain": score_domain,
            "batch_role": batch_role, "transductive": batch_role == "unlabeled_deployment",
            "batch_size": len(rows), "bias_log_scores": bias, "calibrated_scores": calibrated,
            "predictions": [_prediction(row) for row in calibrated], "context_sha256": contexts[0],
            "label_order": labels, "gamma": 1.0, "uses_ground_truth_labels": False}
