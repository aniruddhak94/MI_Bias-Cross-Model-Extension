#!/usr/bin/env python3
"""Evaluate a locked static intervention on a held-out prompt-pair dataset."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.data import load_prompt_pairs
from mi_bias.guardrails import validate_bias_metric_pairs
from mi_bias.interventions import build_residual_hooks
from mi_bias.metrics import score_prompt_bias, summarize_paired_changes
from mi_bias.models import load_hooked_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_data", type=Path,
                        help="Held-out JSON/CSV with male/female continuation lists")
    parser.add_argument("transforms", type=Path, help="Artifact produced by fit_directions.py")
    parser.add_argument("--caa-strength", type=float, default=0.0)
    parser.add_argument("--leace-alpha", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/static_intervention.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    import torch
    transforms = torch.load(args.transforms, map_location="cpu", weights_only=False)
    pairs = load_prompt_pairs(args.test_data)
    validate_bias_metric_pairs(pairs)
    loaded = load_hooked_model(
        transforms["model_name"], revision=transforms.get("model_revision"),
        device=args.device, dtype=transforms.get("dtype", "auto"), token=args.hf_token
    )
    saved_revision = transforms.get("model_revision")
    if saved_revision and loaded.model_revision and saved_revision != loaded.model_revision:
        raise ValueError("Loaded model revision does not match the fitted transform artifact")
    hooks = build_residual_hooks(
        transforms["caa_directions"], transforms["leace_transforms"],
        caa_strength=args.caa_strength, leace_alpha=args.leace_alpha,
        expected_d_model=int(loaded.model.cfg.d_model),
        n_layers=int(loaded.model.cfg.n_layers),
    )
    before, after, per_prompt = [], [], []
    for pair in pairs:
        male = pair.metadata.get("male_continuations")
        female = pair.metadata.get("female_continuations")
        if not isinstance(male, list) or not isinstance(female, list):
            raise ValueError(
                f"Pair {pair.pair_id} requires male_continuations and female_continuations arrays"
            )
        baseline = score_prompt_bias(loaded.model, pair.clean, male, female)
        treated = score_prompt_bias(loaded.model, pair.clean, male, female, hooks=hooks)
        before_value = float(baseline["bias_score"].item())
        after_value = float(treated["bias_score"].item())
        before.append(before_value)
        after.append(after_value)
        per_prompt.append({
            "pair_id": pair.pair_id,
            "clean_prompt": pair.clean,
            "bias_before": before_value,
            "bias_after": after_value,
            "reduction": before_value - after_value,
            "male_logprobs_before": baseline["male_logprobs"].tolist(),
            "female_logprobs_before": baseline["female_logprobs"].tolist(),
            "male_logprobs_after": treated["male_logprobs"].tolist(),
            "female_logprobs_after": treated["female_logprobs"].tolist(),
        })
    result = {
        "model_name": transforms["model_name"],
        "model_revision": transforms.get("model_revision"),
        "tokenizer_revision": transforms.get("tokenizer_revision"),
        "dtype": transforms.get("dtype"),
        "layers": transforms["layers"],
        "caa_strength": args.caa_strength,
        "leace_alpha": args.leace_alpha,
        "summary": summarize_paired_changes(before, after),
        "per_prompt": per_prompt,
        "selection_note": "Parameters must be locked using validation data before this held-out evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved held-out intervention report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
