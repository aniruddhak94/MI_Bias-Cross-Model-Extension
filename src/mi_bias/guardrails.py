"""Preflight checks that catch common data leakage and evaluation mistakes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from .types import PromptPair


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_bias_metric_pairs(pairs: Sequence[PromptPair]) -> None:
    """Validate the matched continuation fields before loading a model."""
    if not pairs:
        raise ValueError("Bias evaluation requires at least one prompt pair")
    for pair in pairs:
        male = pair.metadata.get("male_continuations")
        female = pair.metadata.get("female_continuations")
        for name, values in (("male_continuations", male),
                             ("female_continuations", female)):
            if (not isinstance(values, list) or not values
                    or any(not isinstance(item, str) or not item.strip() for item in values)):
                raise ValueError(
                    f"Pair {pair.pair_id} needs a non-empty list of strings for {name}"
                )


def validate_disjoint_splits(
    splits: Mapping[str, Sequence[PromptPair]], *, min_groups_warning: int = 30
) -> list[str]:
    """Fail on ID/group/text leakage across train, validation, and test.

    Returns informational warnings for small split sizes; small sample counts
    are reported rather than rejected so pilot runs remain possible.
    """
    if len(splits) < 2:
        raise ValueError("Provide at least two named splits for a leakage audit")
    owners: dict[str, dict[str, str]] = {
        "pair_id": {}, "group_id": {}, "prompt_text": {}
    }
    group_counts: dict[str, int] = {}
    for split_name, pairs in splits.items():
        if not pairs:
            raise ValueError(f"Split '{split_name}' is empty")
        group_counts[split_name] = len({pair.group_id for pair in pairs})
        for pair in pairs:
            for kind, value in (("pair_id", pair.pair_id), ("group_id", pair.group_id)):
                prior = owners[kind].get(value)
                if prior is not None and prior != split_name:
                    raise ValueError(
                        f"Leakage: {kind} {value!r} occurs in both '{prior}' and '{split_name}'"
                    )
                owners[kind][value] = split_name
            for prompt in (pair.clean, pair.corrupted):
                normalized = _normalise_text(prompt)
                prior = owners["prompt_text"].get(normalized)
                if prior is not None and prior != split_name:
                    raise ValueError(
                        f"Leakage: identical prompt text occurs in '{prior}' and '{split_name}'"
                    )
                owners["prompt_text"][normalized] = split_name

    return [
        f"Split '{name}' has only {count} unique groups; treat estimates as exploratory."
        for name, count in group_counts.items() if count < min_groups_warning
    ]
