#!/usr/bin/env python3
"""Measure held-out causal-LM perplexity before and after an intervention."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.interventions import build_residual_hooks
from mi_bias.models import load_hooked_model


def sentence_nll(model, text, hooks=None):
    import torch
    import torch.nn.functional as F
    tokens = model.to_tokens(text, prepend_bos=True).to(model.cfg.device)
    if tokens.shape[1] < 2:
        return None, 0
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        n_tokens = int(tokens.shape[1] - 1)
        total = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            tokens[:, 1:].reshape(-1), reduction="sum",
        )
    return float(total.item()), n_tokens


def corpus_summary(records, seed=42, samples=5000):
    import numpy as np
    rng = np.random.default_rng(seed)
    # Token-weighted corpus perplexity; bootstrap examples for a paired interval
    # on the intervention's multiplicative perplexity change.
    before_total = sum(row["nll_before"] for row in records)
    after_total = sum(row["nll_after"] for row in records)
    total_tokens = sum(row["n_tokens"] for row in records)
    before_ppl = float(np.exp(before_total / total_tokens))
    after_ppl = float(np.exp(after_total / total_tokens))
    indices = rng.integers(0, len(records), size=(samples, len(records)))
    before_by_row = np.asarray([row["nll_before"] for row in records])
    after_by_row = np.asarray([row["nll_after"] for row in records])
    tokens_by_row = np.asarray([row["n_tokens"] for row in records])
    sampled_tokens = tokens_by_row[indices].sum(axis=1)
    sampled_log_ratio = (after_by_row[indices].sum(axis=1)
                         - before_by_row[indices].sum(axis=1)) / sampled_tokens
    ratio_ci = np.quantile(np.exp(sampled_log_ratio), [0.025, 0.975])
    return {
        "n_sentences": len(records), "n_tokens": total_tokens,
        "perplexity_before": before_ppl, "perplexity_after": after_ppl,
        "perplexity_ratio_after_over_before": float(after_ppl / before_ppl),
        "perplexity_change_percent": float(100 * (after_ppl / before_ppl - 1)),
        "paired_ratio_bootstrap_95ci": [float(ratio_ci[0]), float(ratio_ci[1])],
        "bootstrap_seed": seed, "bootstrap_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSON list with a 'text' field")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision")
    parser.add_argument("--transforms", type=Path)
    parser.add_argument("--caa-strength", type=float, default=0.0)
    parser.add_argument("--leace-alpha", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output", type=Path, default=Path("results/perplexity.json"))
    parser.add_argument("--dataset-label", default="held-out text")
    args = parser.parse_args()

    import torch
    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or any(
        not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip()
        for row in data
    ):
        raise ValueError("Dataset must be a non-empty JSON list of objects with non-empty 'text'")
    transforms = torch.load(args.transforms, map_location="cpu", weights_only=False) if args.transforms else None
    model_name = transforms["model_name"] if transforms else args.model
    revision = transforms.get("model_revision") if transforms else args.revision
    dtype = transforms.get("dtype", args.dtype) if transforms else args.dtype
    loaded = load_hooked_model(model_name, revision=revision, device=args.device,
                               dtype=dtype, token=args.hf_token)
    hooks = None
    if transforms and not (args.caa_strength == 0 and args.leace_alpha == 0):
        hooks = build_residual_hooks(
            transforms["caa_directions"], transforms["leace_transforms"],
            caa_strength=args.caa_strength, leace_alpha=args.leace_alpha,
            expected_d_model=int(loaded.model.cfg.d_model),
            n_layers=int(loaded.model.cfg.n_layers),
        )
    per_sentence = []
    for index, row in enumerate(data):
        before, count = sentence_nll(loaded.model, row["text"])
        after, treated_count = sentence_nll(loaded.model, row["text"], hooks=hooks)
        if before is None or after is None or count != treated_count:
            continue
        per_sentence.append({
            "id": str(row.get("id", index)), "nll_before": before,
            "nll_after": after, "n_tokens": count,
        })
    result = {
        "model_name": loaded.model_name, "model_revision": loaded.model_revision,
        "tokenizer_revision": loaded.tokenizer_revision, "dtype": loaded.dtype,
        "dataset_label": args.dataset_label,
        "interpretation": "Language-modeling perplexity only; this is not task accuracy or a CoLA MCC/F1 score.",
        "intervention": {"caa_strength": args.caa_strength, "leace_alpha": args.leace_alpha},
        "summary": corpus_summary(per_sentence), "per_sentence": per_sentence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved perplexity report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
