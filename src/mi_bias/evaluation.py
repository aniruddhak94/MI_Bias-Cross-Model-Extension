"""Capability evaluation helpers using task-appropriate metrics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def language_model_perplexity(model, sentences, hooks=None) -> float:
    """Compute token-weighted causal LM perplexity, optionally under hooks."""
    total_loss = 0.0
    total_tokens = 0
    for sentence in sentences:
        tokens = model.to_tokens(sentence, prepend_bos=True).to(model.cfg.device)
        if tokens.shape[1] < 2:
            continue
        with torch.inference_mode():
            logits = (model.run_with_hooks(tokens, fwd_hooks=hooks)
                      if hooks else model(tokens))
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                tokens[:, 1:].reshape(-1), reduction="sum"
            )
        total_loss += float(loss.item())
        total_tokens += int(tokens.shape[1] - 1)
    if total_tokens == 0:
        raise ValueError("No target tokens were available for perplexity evaluation")
    return math.exp(total_loss / total_tokens)


def classification_metrics(labels, predictions) -> dict[str, float]:
    """Return accuracy, Matthews correlation, and macro F1 (including CoLA)."""
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "matthews_correlation": float(matthews_corrcoef(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def entity_span_f1(gold_sequences, predicted_sequences) -> dict[str, float | int]:
    """Compute exact-span micro precision/recall/F1 for BIO-tagged NER output."""
    def spans(tags):
        found = set()
        start, entity_type = None, None
        for index, tag in enumerate([*tags, "O"]):
            prefix, _, kind = tag.partition("-")
            if prefix not in {"B", "I"}:
                prefix, kind = "O", ""
            if prefix == "B" or (prefix == "I" and (start is None or kind != entity_type)):
                if start is not None:
                    found.add((start, index, entity_type))
                start, entity_type = index, kind
            elif prefix == "O" and start is not None:
                found.add((start, index, entity_type))
                start, entity_type = None, None
        if start is not None:
            found.add((start, len(tags), entity_type))
        return found

    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence counts differ")
    gold = {(i, *span) for i, seq in enumerate(gold_sequences) for span in spans(seq)}
    predicted = {(i, *span) for i, seq in enumerate(predicted_sequences) for span in spans(seq)}
    true_positive = len(gold & predicted)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(gold), 1)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "gold_entities": len(gold), "predicted_entities": len(predicted),
    }
