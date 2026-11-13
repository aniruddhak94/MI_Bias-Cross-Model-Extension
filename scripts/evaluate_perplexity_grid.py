#!/usr/bin/env python3
"""Measure validation capability cost for a predeclared intervention grid."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.interventions import build_residual_hooks
from mi_bias.models import load_hooked_model
from evaluate_perplexity import corpus_summary, sentence_nll


def parse_settings(value):
    result = []
    for item in value.split(","):
        caa, alpha = (float(part) for part in item.split(":"))
        if caa < 0 or not 0 <= alpha <= 1:
            raise argparse.ArgumentTypeError("CAA must be >= 0 and LEACE alpha in [0,1]")
        result.append((caa, alpha))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("--settings", type=parse_settings, default=parse_settings("0:0,3:0,0:0.3,3:0.3"))
    parser.add_argument("--output", type=Path, default=Path("results/capability_selection.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    import torch
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip()
        for row in rows
    ):
        raise ValueError("Dataset must be a nonempty JSON list with text fields")
    transforms = torch.load(args.transforms, map_location="cpu", weights_only=False)
    loaded = load_hooked_model(
        transforms["model_name"], revision=transforms.get("model_revision"),
        device=args.device, dtype=transforms.get("dtype", "auto"), token=args.hf_token,
    )
    if transforms.get("model_revision") and loaded.model_revision and transforms["model_revision"] != loaded.model_revision:
        raise ValueError("Loaded model revision does not match fitted transforms")
    shape = {"expected_d_model": int(loaded.model.cfg.d_model),
             "n_layers": int(loaded.model.cfg.n_layers)}
    baseline_rows = []
    for index, row in enumerate(rows):
        nll, count = sentence_nll(loaded.model, row["text"])
        if nll is not None:
            baseline_rows.append({"index": index, "id": str(row.get("id", index)),
                                  "text": row["text"], "nll_before": nll,
                                  "nll_after": nll, "n_tokens": count})
    reports = []
    for caa, alpha in args.settings:
        hooks = None if (caa == 0 and alpha == 0) else build_residual_hooks(
            transforms["caa_directions"], transforms["leace_transforms"],
            caa_strength=caa, leace_alpha=alpha, **shape,
        )
        treated = []
        for base_row in baseline_rows:
            before = base_row["nll_before"]
            count = base_row["n_tokens"]
            after, treated_count = sentence_nll(loaded.model, base_row["text"], hooks=hooks)
            if after is not None and count == treated_count:
                treated.append({"id": base_row["id"], "nll_before": before,
                                "nll_after": after, "n_tokens": count})
        reports.append({"caa_strength": caa, "leace_alpha": alpha,
                        "summary": corpus_summary(treated), "per_sentence": treated})
    result = {
        "model_name": loaded.model_name, "model_revision": loaded.model_revision,
        "tokenizer_revision": loaded.tokenizer_revision, "dtype": loaded.dtype,
        "dataset_label": "CoLA train sample used for validation-time capability selection",
        "interpretation": "Language-modeling perplexity only, not CoLA task accuracy.",
        "settings": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved capability selection report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
