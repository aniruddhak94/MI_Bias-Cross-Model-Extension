#!/usr/bin/env python3
"""Train a prompt sensitivity gate from annotated JSON records."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.gate import fit_prompt_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path,
                        help="JSON list with prompt, bias_relevant (0/1), group_id")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prompt_gate.joblib"))
    args = parser.parse_args()

    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    prompts = [item["prompt"] for item in records]
    labels = [int(item["bias_relevant"]) for item in records]
    groups = [str(item.get("group_id", index)) for index, item in enumerate(records)]
    gate = fit_prompt_gate(
        prompts, labels, groups=groups, seed=args.seed, target_recall=args.target_recall
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(gate, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({
        "threshold": gate.threshold,
        "validation_metrics": gate.validation_metrics,
        "seed": args.seed,
        "target_recall": args.target_recall,
    }, indent=2), encoding="utf-8")
    print(f"Saved prompt gate: {args.output.resolve()}")


if __name__ == "__main__":
    main()
