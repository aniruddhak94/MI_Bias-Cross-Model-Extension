"""Prompt-pair loading, validation, and leakage-safe group splitting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .types import PromptPair


def load_prompt_pairs(path: str | Path) -> list[PromptPair]:
    """Load JSON records or a CSV with ``clean`` and ``corrupted`` columns."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        records = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as stream:
            records = list(csv.DictReader(stream))
    else:
        raise ValueError(f"Expected .json or .csv dataset: {source}")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Dataset must contain a non-empty list: {source}")

    pairs: list[PromptPair] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        clean = record.get("clean", record.get("clean_prompt", record.get("sentence")))
        corrupted = record.get(
            "corrupted", record.get("corrupted_prompt", record.get("control"))
        )
        if not isinstance(clean, str) or not clean.strip():
            raise ValueError(f"Record {index} has no non-empty clean prompt")
        if not isinstance(corrupted, str) or not corrupted.strip():
            raise ValueError(f"Record {index} has no non-empty control prompt")
        pair_id = str(record.get("id", record.get("pair_id", index)))
        if pair_id in seen:
            raise ValueError(f"Duplicate prompt pair id: {pair_id}")
        seen.add(pair_id)
        pairs.append(PromptPair(
            pair_id=pair_id,
            clean=clean,
            corrupted=corrupted,
            group_id=str(record.get("group_id", pair_id)),
            clean_label=record.get("clean_label"),
            corrupted_label=record.get("corrupted_label"),
            bias_type=record.get("bias_type"),
            metadata={k: v for k, v in record.items() if k not in {
                "id", "pair_id", "clean", "clean_prompt", "sentence", "corrupted",
                "corrupted_prompt", "control", "group_id", "clean_label",
                "corrupted_label", "bias_type"
            }},
        ))
    return pairs


def grouped_train_validation_split(
    pairs: Iterable[PromptPair], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[PromptPair], list[PromptPair]]:
    """Split complete prompt groups, keeping each counterfactual pair together."""
    from sklearn.model_selection import GroupShuffleSplit

    items = list(pairs)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(items) < 2:
        raise ValueError("At least two prompt pairs are required for a split")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    train_idx, validation_idx = next(splitter.split(items, groups=[p.group_id for p in items]))
    return ([items[i] for i in train_idx], [items[i] for i in validation_idx])
