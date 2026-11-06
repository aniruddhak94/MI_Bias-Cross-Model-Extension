#!/usr/bin/env python3
"""Compare thesis-reported values with saved run artifacts."""

import argparse
import json
from pathlib import Path


CLAIM_TO_ARTIFACT = {
    "mean_bias_before": ("metrics", "bias_before"),
    "mean_bias_after": ("metrics", "bias_after"),
    "bias_reduction_percent": ("metrics", "bias_reduction_percent"),
    "cola_perplexity_before": ("metrics", "baseline_perplexity"),
    "cola_perplexity_after": ("metrics", "ablated_perplexity"),
    "prompts_improved": ("metrics", "prompts_improved"),
}


def nested_get(value, path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved_run", type=Path, help="Prior hybrid result JSON")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/replication_gpt2_small.json"))
    parser.add_argument("--probe-results", type=Path,
                        help="Optional saved layerwise probe JSON")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    saved = json.loads(args.saved_run.read_text(encoding="utf-8"))
    claims = config.get("options", {}).get("reported_claims", {})
    metrics = saved.get("metrics", saved)

    print(f"Thesis claim comparison: {args.saved_run}")
    for claim, path in CLAIM_TO_ARTIFACT.items():
        reported = claims.get(claim)
        artifact = nested_get({"metrics": metrics}, path)
        if reported is None:
            continue
        if artifact is None:
            print(f"{claim}: reported={reported}; saved artifact has no mapped value")
            continue
        difference = float(artifact) - float(reported)
        print(f"{claim}: reported={reported}; saved={artifact}; delta={difference:+.6g}")

    if args.probe_results:
        probe = json.loads(args.probe_results.read_text(encoding="utf-8"))
        print("Probe summary:", json.dumps({
            "best_layer": probe.get("best_layer"),
            "best_accuracy": probe.get("best_accuracy"),
            "n_layers_probed": probe.get("n_layers_probed"),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
