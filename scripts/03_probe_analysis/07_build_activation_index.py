#!/usr/bin/env python3
"""Build a probe-ready index for saved Scenario 1 activation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.saved_activations import (  # noqa: E402
    build_activation_index,
    load_trajectory_records,
    summarize_activation_index,
    write_activation_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index saved activation checkpoints without loading model weights or "
            "starting GPU compute."
        )
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/scenario1/trajectories/live",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Local equivalent of the Modal artifact-volume root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/activation_index.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/activation_index_summary.json",
    )
    parser.add_argument("--require-local", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.trajectory_root.glob("*/*.json"))
    records = load_trajectory_records(paths)
    rows = build_activation_index(
        records,
        artifact_root=args.artifact_root,
        require_local=args.require_local,
        verify_checksums=args.verify_checksums,
    )
    write_activation_index(rows, args.output)
    summary = summarize_activation_index(rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        **summary,
        "index_path": args.output.as_posix(),
        "summary_path": args.summary_output.as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
