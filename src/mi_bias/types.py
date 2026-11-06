"""Shared data types used by experiment stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptPair:
    """Matched prompts and labels for one experimental unit.

    Labels are intentionally supplied by the dataset rather than inferred from
    ``clean``/``corrupted``. That distinction prevents neutralization controls
    from silently becoming protected-attribute labels.
    """

    pair_id: str
    clean: str
    corrupted: str
    group_id: str
    clean_label: int | str | None = None
    corrupted_label: int | str | None = None
    bias_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated common run configuration loaded from JSON."""

    experiment_name: str
    model_name: str
    model_revision: str | None
    bias_types: tuple[str, ...]
    datasets: tuple[str, ...]
    seed: int = 42
    device: str = "auto"
    output_root: Path = Path("runs")
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentConfig":
        required = ("experiment_name", "model_name", "bias_types", "datasets")
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(f"Missing required config keys: {', '.join(missing)}")
        known = set(required) | {
            "model_revision", "seed", "device", "output_root", "options"
        }
        return cls(
            experiment_name=str(data["experiment_name"]),
            model_name=str(data["model_name"]),
            model_revision=data.get("model_revision"),
            bias_types=tuple(str(v) for v in data["bias_types"]),
            datasets=tuple(str(v) for v in data["datasets"]),
            seed=int(data.get("seed", 42)),
            device=str(data.get("device", "auto")),
            output_root=Path(data.get("output_root", "runs")),
            options={**data.get("options", {}), **{
                k: v for k, v in data.items() if k not in known
            }},
        )
