#!/usr/bin/env python3
"""Create an auditable run directory from a JSON experiment configuration."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mi_bias.runs import create_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Versioned experiment JSON config")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to hash; repeat as needed")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = create_run_directory(config, args.dataset, args.output_root)
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
