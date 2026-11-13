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
from mi_bias.metrics import (
    counterfactual_stereotype_effect, score_prompt_bias, score_thesis_legacy_l2,
    summarize_paired_changes, thesis_gender_token_ids,
)
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
    hooks = None if args.caa_strength == 0 and args.leace_alpha == 0 else build_residual_hooks(
        transforms["caa_directions"], transforms["leace_transforms"],
        caa_strength=args.caa_strength, leace_alpha=args.leace_alpha,
        expected_d_model=int(loaded.model.cfg.d_model),
        n_layers=int(loaded.model.cfg.n_layers),
    )
    before, after, counter_before, counter_after = [], [], [], []
    legacy_before, legacy_after, per_prompt = [], [], []
    thesis_token_ids = thesis_gender_token_ids(loaded.model)
    for pair in pairs:
        male = pair.metadata.get("male_continuations")
        female = pair.metadata.get("female_continuations")
        if not isinstance(male, list) or not isinstance(female, list):
            raise ValueError(
                f"Pair {pair.pair_id} requires male_continuations and female_continuations arrays"
            )
        baseline = score_prompt_bias(loaded.model, pair.clean, male, female)
        control_baseline = score_prompt_bias(loaded.model, pair.corrupted, male, female)
        treated = score_prompt_bias(loaded.model, pair.clean, male, female, hooks=hooks)
        control_treated = score_prompt_bias(
            loaded.model, pair.corrupted, male, female, hooks=hooks
        )
        before_value = float(baseline["bias_score"].item())
        after_value = float(treated["bias_score"].item())
        before.append(before_value)
        after.append(after_value)
        counter_before_metrics = counterfactual_stereotype_effect(
            baseline, control_baseline, pair.metadata.get("expected_bias_direction")
        )
        counter_after_metrics = counterfactual_stereotype_effect(
            treated, control_treated, pair.metadata.get("expected_bias_direction")
        )
        counter_before.append(counter_before_metrics["stereotype_effect_absolute"])
        counter_after.append(counter_after_metrics["stereotype_effect_absolute"])
        legacy_before_value = score_thesis_legacy_l2(
            loaded.model, pair.clean, *thesis_token_ids
        )
        legacy_after_value = score_thesis_legacy_l2(
            loaded.model, pair.clean, *thesis_token_ids, hooks=hooks
        )
        legacy_before.append(legacy_before_value)
        legacy_after.append(legacy_after_value)
        per_prompt.append({
            "pair_id": pair.pair_id,
            "group_id": pair.group_id,
            "clean_prompt": pair.clean,
            "bias_before": before_value,
            "bias_after": after_value,
            "reduction": before_value - after_value,
            "control_bias_before": float(control_baseline["bias_score"].item()),
            "control_bias_after": float(control_treated["bias_score"].item()),
            "counterfactual_bias_before": counter_before_metrics["stereotype_effect_absolute"],
            "counterfactual_bias_after": counter_after_metrics["stereotype_effect_absolute"],
            "counterfactual_signed_before": counter_before_metrics["stereotype_effect_signed"],
            "counterfactual_signed_after": counter_after_metrics["stereotype_effect_signed"],
            "thesis_legacy_l2_before": legacy_before_value,
            "thesis_legacy_l2_after": legacy_after_value,
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
        "summary": summarize_paired_changes(
            counter_before, counter_after, groups=[pair.group_id for pair in pairs]
        ),
        "clean_prompt_summary": summarize_paired_changes(
            before, after, groups=[pair.group_id for pair in pairs]
        ),
        "thesis_legacy_l2_summary": summarize_paired_changes(
            legacy_before, legacy_after, groups=[pair.group_id for pair in pairs]
        ),
        "per_prompt": per_prompt,
        "primary_metric": "Counterfactual expected-direction effect: absolute change in gender log-odds on occupation prompt relative to its neutralized control.",
        "selection_note": "Parameters must be locked using validation data before this held-out evaluation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved held-out intervention report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
