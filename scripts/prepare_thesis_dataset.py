#!/usr/bin/env python3
"""Prepare the original 200-pair thesis gender set for group-held-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.model_selection import train_test_split

from mi_bias.data import load_prompt_pairs
from mi_bias.guardrails import validate_disjoint_splits


# Same first-token vocabulary and order used by the previous thesis L2 metric.
MALE_WORDS = [
    "he", "him", "his", "man", "men", "boy", "boys", "male", "father", "son",
    "husband", "brother", "gentleman", "sir", "king", "prince", "himself", "mr",
    "businessman", "uncle",
]
FEMALE_WORDS = [
    "she", "her", "hers", "woman", "women", "girl", "girls", "female", "mother",
    "daughter", "wife", "sister", "lady", "madam", "queen", "princess", "herself",
    "mrs", "businesswoman", "aunt",
]
MALE_CONTINUATIONS = [" he", " him", " his"]
FEMALE_CONTINUATIONS = [" she", " her", " hers"]


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not value:
        raise ValueError("Cannot create an empty occupation group ID")
    return value


def prepare(source: Path, output: Path, seed: int) -> dict:
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    if len(original) != 200:
        raise ValueError(f"Expected the original 200-pair thesis dataset, found {len(original)}")
    if len(MALE_WORDS) != len(FEMALE_WORDS):
        raise ValueError("Historical male/female token lists must be aligned")

    rows = []
    seen_ids = set()
    for item in original:
        required = ("id", "clean_prompt", "corrupted_prompt", "biased_word", "expected_bias_direction")
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise ValueError(f"{item.get('id', '<missing id>')} is missing {missing}")
        pair_id = str(item["id"])
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate thesis pair ID: {pair_id}")
        seen_ids.add(pair_id)
        direction = str(item["expected_bias_direction"]).casefold()
        if direction not in {"male", "female"}:
            raise ValueError(f"Unexpected bias direction {direction!r} for {pair_id}")
        group = f"occupation_{slug(str(item['biased_word']))}"
        rows.append({
            "id": pair_id,
            "group_id": group,
            "bias_type": item.get("bias_type", "gender_occupation"),
            "clean": item["clean_prompt"],
            "corrupted": item["corrupted_prompt"],
            "clean_label": 1,
            "corrupted_label": 0,
            "male_continuations": MALE_CONTINUATIONS,
            "female_continuations": FEMALE_CONTINUATIONS,
            "bias_relevant": 1,
            "expected_bias_direction": direction,
            "biased_word": item["biased_word"],
            "neutral_word": item.get("neutral_word"),
            "label_note": "clean is the stereotype-associated thesis prompt; corrupted is its neutralized control",
        })

    # Split the occupation groups themselves with stereotype direction
    # stratification; then map all prompt variants back to their group split.
    group_directions = {}
    for row in rows:
        prior = group_directions.setdefault(row["group_id"], row["expected_bias_direction"])
        if prior != row["expected_bias_direction"]:
            raise ValueError(f"Occupation group {row['group_id']} has conflicting directions")
    groups = sorted(group_directions)
    directions = [group_directions[group] for group in groups]
    trainval_groups, test_groups = train_test_split(
        groups, test_size=0.20, stratify=directions, random_state=seed
    )
    trainval_direction = [group_directions[group] for group in trainval_groups]
    train_groups, val_groups = train_test_split(
        trainval_groups, test_size=0.25, stratify=trainval_direction, random_state=seed + 1
    )
    splits = {
        "train": [row for row in rows if row["group_id"] in set(train_groups)],
        "validation": [row for row in rows if row["group_id"] in set(val_groups)],
        "test": [row for row in rows if row["group_id"] in set(test_groups)],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        (output / f"{name}.json").write_text(json.dumps(split_rows, indent=2), encoding="utf-8")

    gate_rows = []
    for row in splits["train"]:
        gate_rows.extend([
            {"prompt": row["clean"], "bias_relevant": 1, "group_id": row["group_id"]},
            {"prompt": row["corrupted"], "bias_relevant": 0, "group_id": row["group_id"]},
        ])
    (output / "gate_train_annotations.json").write_text(
        json.dumps(gate_rows, indent=2), encoding="utf-8"
    )

    # Verify all splits and scoring fields before a model is downloaded.
    loaded = {name: load_prompt_pairs(output / f"{name}.json") for name in splits}
    warnings = validate_disjoint_splits(loaded, min_groups_warning=30)
    for split in loaded.values():
        for pair in split:
            male = pair.metadata["male_continuations"]
            female = pair.metadata["female_continuations"]
            if len(male) != len(female) or len(male) != len(MALE_CONTINUATIONS):
                raise ValueError(f"Unpaired candidate continuations in {pair.pair_id}")

    manifest = {
        "dataset": "original 200-pair thesis gender_occupation set",
        "source_file": source.name,
        "source_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "seed": seed,
        "group_key": "biased_word (occupation); all prompt variants remain in one split",
        "stratification": "occupation groups stratified by expected_bias_direction (male/female)",
        "target_split_fractions": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "label_meaning": {"1": "stereotype-associated thesis prompt", "0": "neutralized paired control"},
        "primary_continuations": "Three matched pronoun continuations (he/she, him/her, his/hers), scored with complete token-aware continuation likelihoods.",
        "historical_metric": "The original 20-word male/female vocabularies are evaluated separately with the previous thesis first-token L2 metric.",
        "splits": {
            name: {
                "pairs": len(part),
                "groups": len({row["group_id"] for row in part}),
                "male_direction": sum(row["expected_bias_direction"] == "male" for row in part),
                "female_direction": sum(row["expected_bias_direction"] == "female" for row in part),
            }
            for name, part in splits.items()
        },
        "warnings": warnings,
        "gate_scope": "Matched thesis prompt/control pairs only; this is not a broadly validated context detector.",
        "limitations": [
            "Occupation is the grouping unit; held-out results generalize to unseen occupations in this dataset, not arbitrary domains.",
            "The held-out test set contains about 40 pairs; uncertainty and limited domain coverage must be reported.",
            "Historical full-200 thesis figures are a replication reference, not a held-out estimate.",
        ],
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['pairs']} pairs, {info['groups']} occupation groups, "
              f"direction male/female={info['male_direction']}/{info['female_direction']}")
    for warning in warnings:
        print("WARNING:", warning)
    print("Saved dataset manifest:", output / "dataset_manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=PROJECT_ROOT / "Dataset" / "thesis_gender_bias_200.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "kaggle_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
