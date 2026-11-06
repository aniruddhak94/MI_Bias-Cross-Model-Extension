"""Reproducible run directories and metadata manifests."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(cwd: str | Path = ".") -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty(cwd: str | Path = ".") -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, check=True,
            capture_output=True, text=True
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "transformers", "transformer-lens", "scikit-learn", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def create_run_directory(config: dict[str, Any], dataset_paths=(), root="runs") -> Path:
    """Create an isolated UTC-stamped run with config and audit manifest."""
    now = datetime.now(timezone.utc)
    experiment_slug = re.sub(
        r"[^A-Za-z0-9_.-]+", "-", str(config.get("experiment_name", "experiment"))
    ).strip("-.") or "experiment"
    run_id = f"{now:%Y%m%dT%H%M%SZ}_{experiment_slug}"
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "created_at_utc": now.isoformat(),
        "code_revision": git_revision(),
        "working_tree_dirty": git_is_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "model_name": config.get("model_name"),
        "model_revision": config.get("model_revision"),
        "seed": config.get("seed", 42),
        "datasets": {
            str(Path(path)): sha256_file(path) for path in dataset_paths if Path(path).is_file()
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    for name in ("data", "metrics", "circuits", "interventions", "evaluation", "figures"):
        (run_dir / name).mkdir()
    return run_dir
