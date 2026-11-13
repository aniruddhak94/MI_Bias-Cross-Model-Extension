#!/usr/bin/env python3
"""Run the staged GPT-2 small / GPT-2 Large / Pythia-1B Kaggle workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "gpt2_small": {"label": "GPT-2 small", "name": "gpt2", "dtype": "float32", "layers": "6,8,9,10,11"},
    "gpt2_large": {"label": "GPT-2 Large", "name": "gpt2-large", "dtype": "float16", "layers": "15,21,27,32,35"},
    "pythia_1b": {"label": "Pythia-1B", "name": "EleutherAI/pythia-1b", "dtype": "float32", "layers": "3,6,9,12,15"},
}
GRID = [(0.0, 0.0), (3.0, 0.0), (0.0, 0.3), (3.0, 0.3)]
MAX_VALIDATION_PPL_INCREASE_PERCENT = 10.15
MODEL_ESTIMATE_HOURS = {
    "gpt2_small": (0.5, 1.25),
    "gpt2_large": (1.0, 2.5),
    "pythia_1b": (0.75, 1.75),
}


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("\nRUN:", " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"Completed in {(time.monotonic() - started) / 60:.1f} min", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("kaggle_data"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/cross_model_runs"))
    parser.add_argument("--run-id", help="Reuse this ID to resume an earlier run")
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()

    slugs = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown = sorted(set(slugs) - set(MODELS))
    if unknown:
        parser.error(f"Unknown model keys {unknown}; choose from {sorted(MODELS)}")
    run_started = time.monotonic()
    low = sum(MODEL_ESTIMATE_HOURS[slug][0] for slug in slugs)
    high = sum(MODEL_ESTIMATE_HOURS[slug][1] for slug in slugs)
    print(f"Planning estimate for {len(slugs)} selected model(s): ~{low:.1f}-{high:.1f} hours on one Kaggle T4; "
          "includes baseline, fitting, validation/capability selection, held-out/gate evaluation, "
          "and CoLA validation perplexity. Actual time depends on downloads and Kaggle load.", flush=True)
    data = args.data_dir.resolve()
    required = [data / "train.json", data / "validation.json", data / "test.json",
                data / "gate_train_annotations.json", data / "cola_capability_selection_text.json",
                data / "cola_validation_text.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing Kaggle data files: " + ", ".join(missing))

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root.resolve() / run_id
    code_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=False
    ).stdout.strip() or None
    state_path = root / "run_state.json"
    if root.exists() and any(root.iterdir()) and not state_path.exists():
        parser.error(f"Run directory {root} has outputs but no protocol state; use a fresh --run-id to avoid mixing runs")
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if (previous.get("code_revision") != code_revision
                or previous.get("protocol") != "counterfactual-capability-budget-v1"
                or previous.get("models") != slugs):
            parser.error("This run ID belongs to different code/protocol. Start a fresh run ID instead of mixing cached outputs.")
    root.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "run_id": run_id, "code_revision": code_revision,
        "protocol": "counterfactual-capability-budget-v1",
        "models": slugs,
    }, indent=2), encoding="utf-8")
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
        candidate_bias = []
        for (caa, alpha), path in zip(GRID, validation_paths):
            report = json.loads(path.read_text(encoding="utf-8"))
            candidate_bias.append((float(report["summary"]["relative_reduction_percent"]), caa, alpha))
        capability_selection = results / f"{slug}_capability_selection.json"
        if not capability_selection.exists():
            run("scripts/evaluate_perplexity_grid.py", data / "cola_capability_selection_text.json",
                direction, "--settings", ",".join(f"{c:g}:{a:g}" for c, a in GRID),
                "--output", capability_selection)
        cap_report = json.loads(capability_selection.read_text(encoding="utf-8"))
        cap_by_setting = {
            (float(row["caa_strength"]), float(row["leace_alpha"])):
                float(row["summary"]["perplexity_change_percent"])
            for row in cap_report["settings"]
        }
        candidates = []
        for bias_reduction, caa, alpha in candidate_bias:
            ppl_change = cap_by_setting[(caa, alpha)]
            if bias_reduction > 0 and ppl_change <= MAX_VALIDATION_PPL_INCREASE_PERCENT:
                candidates.append((bias_reduction, caa, alpha, ppl_change))
        if candidates:
            best = max(candidates, key=lambda row: row[0])
            chosen = {"caa_strength": best[1], "leace_alpha": best[2],
                      "validation_relative_reduction_percent": best[0],
                      "validation_cola_perplexity_change_percent": best[3],
                      "selection_reason": "largest positive counterfactual reduction within capability budget"}
        else:
            chosen = {"caa_strength": 0.0, "leace_alpha": 0.0,
                      "validation_relative_reduction_percent": next(
                          row[0] for row in candidate_bias if row[1:] == (0.0, 0.0)),
                      "validation_cola_perplexity_change_percent": cap_by_setting[(0.0, 0.0)],
                      "selection_reason": "no candidate met both positive-bias-reduction and capability-budget criteria; selected no-op"}
        chosen["selection_rule"] = (
            f"validation counterfactual reduction > 0 and CoLA train-sample perplexity change <= "
            f"{MAX_VALIDATION_PPL_INCREASE_PERCENT:.2f}%; otherwise no-op"
        )
        resolved[slug] = chosen
        print(f"{slug} validation bias candidates: {candidate_bias}; CoLA PPL change: {cap_by_setting}; selected: {chosen}")

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
        if not capability.exists():
            run("scripts/evaluate_perplexity.py", cola_path, "--transforms", direction,
                *common, "--dataset-label", "CoLA validation perplexity", "--output", capability)

    config = {
        "run_id": run_id, "models": {slug: MODELS[slug] for slug in slugs},
        "locked_settings": resolved,
        "data_dir": str(data),
        "dataset_manifest": (json.loads((data / "dataset_manifest.json").read_text(encoding="utf-8"))
                             if (data / "dataset_manifest.json").exists() else None),
        "code_revision": code_revision,
        "environment": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
        },
        "metric_protocol": {
            "primary_bias_metric": "paired counterfactual stereotype effect: signed gender continuation log odds on occupation prompt minus its neutral control, oriented by expected occupation direction; magnitude summarized",
            "historical_bias_metric": "prior thesis's first-token L2 over 20 aligned male/female vocabulary entries",
            "confidence_intervals": "occupation-group bootstrap",
            "validation_selection": f"positive counterfactual bias reduction subject to <= {MAX_VALIDATION_PPL_INCREASE_PERCENT:.2f}% CoLA train-sample perplexity increase; no-op if none qualify",
        },
        "historical_thesis_reference": {
            "gpt2_small_full_200_bias_reduction_percent": 10.07,
            "gpt2_small_prompts": 200,
            "status": "reference only; full-set prior score is not a held-out estimate",
        },
        "analysis_scope": {
            "included": ["group-held-out probes", "CAA/LEACE validation sweep including no-op", "capability-budgeted model-specific selection", "static and gated held-out evaluation", "CoLA validation perplexity"],
            "not_included": ["EAP edge ranking and direct activation-patching validation", "broad independently annotated gate evaluation"],
        },
        "notes": ["Held-out results use about 40 prompts from 20 occupations; generalization is limited to this occupation-prompt domain.",
                  "Capability is CoLA validation language-modeling perplexity, not task accuracy."],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run("scripts/plot_cross_model_results.py", "--results-dir", results,
        "--artifacts-dir", artifacts, "--output-dir", figures,
        "--models", ",".join(slugs), "--labels", ",".join(MODELS[slug]["label"] for slug in slugs))
    print(f"\nCompleted stages for {len(slugs)} models. Run ID: {run_id}")
    print(f"Total elapsed: {(time.monotonic() - run_started) / 3600:.2f} h")
    print(f"Results: {root}")
    print("If interrupted, rerun this command with --run-id", run_id)


if __name__ == "__main__":
    main()
