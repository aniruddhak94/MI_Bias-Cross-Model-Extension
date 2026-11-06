#!/usr/bin/env python3
"""Measure tokenization-aware matched-continuation bias on a dataset."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.data import load_prompt_pairs
from mi_bias.metrics import bootstrap_mean_interval, score_prompt_bias
from mi_bias.models import architecture_metadata, load_hooked_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prompt pairs with candidate continuation arrays")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output", type=Path, default=Path("results/baseline_bias.json"))
    args = parser.parse_args()

    pairs = load_prompt_pairs(args.dataset)
    loaded = load_hooked_model(args.model, revision=args.revision, device=args.device,
                               dtype=args.dtype,
                               token=args.hf_token)
    scores, rows = [], []
    for pair in pairs:
        male = pair.metadata.get("male_continuations")
        female = pair.metadata.get("female_continuations")
        if not isinstance(male, list) or not isinstance(female, list):
            raise ValueError(f"Pair {pair.pair_id} is missing candidate continuation arrays")
        score = score_prompt_bias(loaded.model, pair.clean, male, female)
        value = float(score["bias_score"].item())
        scores.append(value)
        rows.append({
            "pair_id": pair.pair_id,
            "group_id": pair.group_id,
            "bias_type": pair.bias_type,
            "prompt": pair.clean,
            "bias_score": value,
            "male_logprobs": score["male_logprobs"].tolist(),
            "female_logprobs": score["female_logprobs"].tolist(),
        })
    result = {
        "model_name": loaded.model_name,
        "model_revision": loaded.model_revision,
        "tokenizer_revision": loaded.tokenizer_revision,
        "dtype": loaded.dtype,
        "architecture": architecture_metadata(loaded.model),
        "dataset": str(args.dataset),
        "n_prompts": len(scores),
        "bias_summary": {
            "mean_bias": sum(scores) / len(scores),
            "mean_bias_bootstrap_95ci": bootstrap_mean_interval(scores),
            "n": len(scores),
        },
        "per_prompt": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved baseline report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
