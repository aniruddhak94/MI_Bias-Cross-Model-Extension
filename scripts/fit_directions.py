#!/usr/bin/env python3
"""Fit group-safe layer probes, CAA vectors, and LEACE transforms on train data."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.data import grouped_train_validation_split, load_prompt_pairs
from mi_bias.models import architecture_metadata, load_hooked_model
from mi_bias.probes import (
    collect_pair_activations,
    compute_caa_direction,
    compute_leace_transform,
    fit_grouped_probe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_data", type=Path,
                        help="JSON/CSV prompt pairs with explicit binary labels")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--revision")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token; defaults to HF_TOKEN environment variable")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--layers", default="6,8,9,10,11")
    parser.add_argument("--caa-positive-label", default="1",
                        help="Label value whose mean direction CAA will subtract")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/directions.pt"))
    args = parser.parse_args()

    pairs = load_prompt_pairs(args.train_data)
    train_pairs, heldout_pairs = grouped_train_validation_split(pairs, seed=args.seed)
    loaded = load_hooked_model(args.model, revision=args.revision, device=args.device,
                               dtype=args.dtype,
                               token=args.hf_token)
    layers = [int(item) for item in args.layers.split(",")]
    if not layers:
        raise ValueError("At least one layer must be selected")
    if len(set(layers)) != len(layers):
        raise ValueError("Layer list contains duplicates")
    max_layer = int(loaded.model.cfg.n_layers) - 1
    if any(layer < 0 or layer > max_layer for layer in layers):
        raise ValueError(f"Layer ids must be in [0, {max_layer}]")
    representations = collect_pair_activations(loaded.model, train_pairs, layers)
    label_values = representations[layers[0]].label_values
    try:
        caa_positive_label = label_values.index(args.caa_positive_label)
    except ValueError as error:
        raise ValueError(
            f"CAA positive label {args.caa_positive_label!r} not found in {label_values}"
        ) from error

    artifacts = {
        "model_name": args.model,
        "model_revision": loaded.model_revision,
        "tokenizer_revision": loaded.tokenizer_revision,
        "dtype": loaded.dtype,
        "architecture": architecture_metadata(loaded.model),
        "seed": args.seed,
        "layers": layers,
        "label_values": label_values,
        "caa_positive_label": args.caa_positive_label,
        "caa_directions": {},
        "leace_transforms": {},
        "probe_metrics": {},
        "n_train_pairs": len(train_pairs),
        "n_group_heldout_pairs": len(heldout_pairs),
    }
    for layer, data in representations.items():
        _, _, probe_metrics = fit_grouped_probe(data, seed=args.seed)
        matrix, bias = compute_leace_transform(data.activations, data.labels)
        artifacts["caa_directions"][layer] = compute_caa_direction(
            data, positive_label=caa_positive_label
        )
        artifacts["leace_transforms"][layer] = (matrix, bias)
        artifacts["probe_metrics"][layer] = probe_metrics

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(artifacts, args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps({
        key: value for key, value in artifacts.items()
        if key not in {"caa_directions", "leace_transforms"}
    }, indent=2), encoding="utf-8")
    print(f"Saved fitted transforms: {args.output.resolve()}")
    print(f"Saved probe summary: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
