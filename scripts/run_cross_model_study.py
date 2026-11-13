#!/usr/bin/env python3
"""Run the staged GPT-2 small / GPT-2 Large / Llama-2 Kaggle workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gpt2_small": {"label": "GPT-2 small", "name": "gpt2", "dtype": "float32", "layers": "6,8,9,10,11"},
    "gpt2_large": {"label": "GPT-2 Large", "name": "gpt2-large", "dtype": "float16", "layers": "15,21,27,32,35"},
    "llama2_7b": {"label": "Llama-2 7B", "name": "meta-llama/Llama-2-7b-hf", "dtype": "float16", "layers": "8,16,23,28,31"},
}
GRID = [(3.0, 0.0), (0.0, 0.3), (3.0, 0.3)]


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("kaggle_data"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/cross_model_runs"))
    parser.add_argument("--run-id", help="Reuse this ID to resume an earlier run")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--skip-capability", action="store_true")
    args = parser.parse_args()

    slugs = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown = sorted(set(slugs) - set(MODELS))
    if unknown:
        parser.error(f"Unknown model keys {unknown}; choose from {sorted(MODELS)}")
    data = args.data_dir.resolve()
    required = [data / "train.json", data / "validation.json", data / "test.json",
                data / "gate_train_annotations.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing Kaggle data files: " + ", ".join(missing))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root.resolve() / run_id
    results, artifacts, figures = root / "results", root / "artifacts", root / "figures"
    for folder in (results, artifacts, figures):
        folder.mkdir(parents=True, exist_ok=True)
    resolved = {}

    # Catch split leakage before downloading model checkpoints.
    run("scripts/validate_splits.py", "--train", data / "train.json",
        "--validation", data / "validation.json", "--test", data / "test.json")

    for slug in slugs:
        spec = MODELS[slug]
        direction = artifacts / f"{slug}_directions.pt"
        baseline = results / f"{slug}_baseline_test.json"
        if not baseline.exists():
            run("scripts/measure_bias_baseline.py", data / "test.json", "--model", spec["name"],
                "--dtype", spec["dtype"], "--output", baseline)
        if not direction.exists():
            run("scripts/fit_directions.py", data / "train.json", "--model", spec["name"],
                "--dtype", spec["dtype"], "--seed", "42", "--layers", spec["layers"],
                "--output", direction)

        prefix = f"validation_{slug}"
        validation_paths = [results / f"{prefix}_caa_{c:g}_leace_{a:g}.json" for c, a in GRID]
        if not all(path.exists() for path in validation_paths):
            run("scripts/evaluate_static_grid.py", data / "validation.json", direction,
                "--settings", ",".join(f"{c:g}:{a:g}" for c, a in GRID),
                "--output-dir", results, "--prefix", prefix)
        candidates = []
        for (caa, alpha), path in zip(GRID, validation_paths):
            report = json.loads(path.read_text(encoding="utf-8"))
            candidates.append((float(report["summary"]["relative_reduction_percent"]), caa, alpha))
        best = max(candidates, key=lambda row: row[0])
        resolved[slug] = {"caa_strength": best[1], "leace_alpha": best[2],
                          "validation_relative_reduction_percent": best[0],
                          "selection_rule": "maximum mean relative bias reduction on validation"}
        print(f"{slug} validation candidates (relative %, CAA, LEACE): {candidates}; selected: {best}")

    gate = artifacts / "prompt_gate.joblib"
    if not gate.exists():
        run("scripts/train_prompt_gate.py", data / "gate_train_annotations.json",
            "--target-recall", "0.95", "--seed", "42", "--output", gate)

    for slug in slugs:
        direction = artifacts / f"{slug}_directions.pt"
        static = results / f"{slug}_static_test.json"
        gated = results / f"{slug}_gated_test.json"
        setting = resolved[slug]
        common = ["--caa-strength", setting["caa_strength"],
                  "--leace-alpha", setting["leace_alpha"]]
        if not static.exists():
            run("scripts/evaluate_static_intervention.py", data / "test.json", direction,
                *common, "--output", static)
        if not gated.exists():
            run("scripts/evaluate_gated_intervention.py", data / "test.json", direction, gate,
                *common, "--output", gated)
        cola_path = data / "cola_validation_text.json"
        capability = results / f"{slug}_capability.json"
        if cola_path.exists() and not capability.exists() and not args.skip_capability:
            run("scripts/evaluate_perplexity.py", cola_path, "--transforms", direction,
                *common, "--dataset-label", "CoLA validation perplexity", "--output", capability)

    config = {
        "run_id": run_id, "models": {slug: MODELS[slug] for slug in slugs},
        "locked_settings": resolved,
        "data_dir": str(data),
        "notes": ["Starter profession prompts are not thesis replication data.",
                  "Capability is CoLA validation language-modeling perplexity, not task accuracy."],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run("scripts/plot_cross_model_results.py", "--results-dir", results,
        "--artifacts-dir", artifacts, "--output-dir", figures,
        "--models", ",".join(slugs), "--labels", ",".join(MODELS[slug]["label"] for slug in slugs))
    print(f"\nCompleted stages for {len(slugs)} models. Run ID: {run_id}")
    print(f"Results: {root}")
    print("If interrupted, rerun this command with --run-id", run_id)


if __name__ == "__main__":
    main()
