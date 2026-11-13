#!/usr/bin/env python3
"""Preflight audit for train/validation/test prompt split leakage."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.data import load_prompt_pairs
from mi_bias.guardrails import validate_disjoint_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--min-groups-warning", type=int, default=30)
    args = parser.parse_args()

    splits = {
        "train": load_prompt_pairs(args.train),
        "validation": load_prompt_pairs(args.validation),
        "test": load_prompt_pairs(args.test),
    }
    warnings = validate_disjoint_splits(
        splits, min_groups_warning=args.min_groups_warning
    )
    for name, pairs in splits.items():
        print(f"{name}: {len(pairs)} pairs, {len({p.group_id for p in pairs})} groups")
    for warning in warnings:
        print(f"WARNING: {warning}")
    print("PASS: no pair ID, group, or exact normalized prompt leakage across splits")


if __name__ == "__main__":
    main()
