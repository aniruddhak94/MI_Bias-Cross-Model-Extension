#!/usr/bin/env python3
"""Evaluate a small predeclared static-intervention grid with one model load."""

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


def parse_settings(value):
    pairs = []
    for item in value.split(","):
        try:
            caa, alpha = (float(part) for part in item.split(":"))
        except Exception as exc:
            raise argparse.ArgumentTypeError(
                "Use comma-separated CAA:LEACE pairs, e.g. 3:0,0:0.3,3:0.3"
            ) from exc
        if caa < 0 or not 0 <= alpha <= 1:
            raise argparse.ArgumentTypeError("CAA must be >= 0 and LEACE alpha in [0, 1]")
        pairs.append((caa, alpha))
    if not pairs:
        raise argparse.ArgumentTypeError("At least one setting is required")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Validation prompt pairs")
    parser.add_argument("transforms", type=Path, help="Fitted model-specific .pt artifact")
    parser.add_argument("--settings", type=parse_settings, default=parse_settings("3:0,0:0.3,3:0.3"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--prefix", default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    import torch
    transforms = torch.load(args.transforms, map_location="cpu", weights_only=False)
    pairs = load_prompt_pairs(args.data)
    validate_bias_metric_pairs(pairs)
    loaded = load_hooked_model(
        transforms["model_name"], revision=transforms.get("model_revision"),
        device=args.device, dtype=transforms.get("dtype", "auto"), token=args.hf_token,
    )
    fitted_revision = transforms.get("model_revision")
    if fitted_revision and loaded.model_revision and fitted_revision != loaded.model_revision:
        raise ValueError("Loaded model revision does not match fitted transforms")
    shape = {"expected_d_model": int(loaded.model.cfg.d_model),
             "n_layers": int(loaded.model.cfg.n_layers)}
    hooks_grid = [
        build_residual_hooks(
            transforms["caa_directions"], transforms["leace_transforms"],
            caa_strength=caa, leace_alpha=alpha, **shape,
        )
        for caa, alpha in args.settings
    ]
    before = [[] for _ in args.settings]
    after = [[] for _ in args.settings]
    rows = [[] for _ in args.settings]
    for pair in pairs:
        male = pair.metadata["male_continuations"]
        female = pair.metadata["female_continuations"]
        base = score_prompt_bias(loaded.model, pair.clean, male, female)
        base_value = float(base["bias_score"].item())
        for index, hooks in enumerate(hooks_grid):
            treated = score_prompt_bias(loaded.model, pair.clean, male, female, hooks=hooks)
            treated_value = float(treated["bias_score"].item())
            before[index].append(base_value)
            after[index].append(treated_value)
            rows[index].append({
                "pair_id": pair.pair_id, "bias_before": base_value,
                "bias_after": treated_value, "reduction": base_value - treated_value,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (caa, alpha), before_values, after_values, per_prompt in zip(
        args.settings, before, after, rows
    ):
        output = args.output_dir / (
            f"{args.prefix}_caa_{caa:g}_leace_{alpha:g}.json"
        )
        result = {
            "model_name": transforms["model_name"],
            "model_revision": transforms.get("model_revision"),
            "tokenizer_revision": transforms.get("tokenizer_revision"),
            "dtype": transforms.get("dtype"), "layers": transforms["layers"],
            "caa_strength": caa, "leace_alpha": alpha,
            "summary": summarize_paired_changes(before_values, after_values),
            "per_prompt": per_prompt,
            "evaluation_split": "validation",
            "selection_note": "Use validation only for selection; reserve test prompts for one locked evaluation.",
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved validation result: {output.resolve()}")


if __name__ == "__main__":
    main()
