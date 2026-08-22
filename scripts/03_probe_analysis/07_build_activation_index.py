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
    ACTIVATION_INDEX_ANALYSIS_TIERS,
    build_activation_index,
    load_activation_index,
    load_trajectory_records,
    summarize_activation_index,
    trajectory_paths_for_analysis_tier,
    upgrade_legacy_activation_index,
    write_activation_index,
)
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    select_paper_trajectory_records,
    validate_paper_analysis_inputs,
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
        default=PROJECT_ROOT / "experiments/scenario1/outputs/trajectories/live",
    )
    parser.add_argument(
        "--legacy-index",
        type=Path,
        help=(
            "Upgrade a historical v2 index to explicit v3/unclassified metadata "
            "instead of rebuilding from trajectory files."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Local equivalent of the Modal artifact-volume root.",
    )
    parser.add_argument(
        "--analysis-tier",
        required=True,
        choices=sorted(ACTIVATION_INDEX_ANALYSIS_TIERS),
        help=(
            "Select exactly one tiered live namespace; use 'unclassified' only "
            "for historical tierless trajectories."
        ),
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
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
        help=(
            "Tracked protocol-selection policy. Managed historical trajectories "
            "are excluded before the activation index is built."
        ),
    )
    parser.add_argument("--require-local", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    args = parser.parse_args()

    paper_policy = load_paper_input_policy(args.paper_input_policy)
    if args.legacy_index:
        if args.analysis_tier != "unclassified":
            raise ValueError(
                "A legacy activation index can only be upgraded as unclassified."
            )
        if args.require_local or args.verify_checksums:
            raise ValueError(
                "Legacy-index migration only updates metadata; run the downstream "
                "scan without --skip-checksums to verify every activation artifact."
            )
        rows = upgrade_legacy_activation_index(load_activation_index(args.legacy_index))
        paper_input_selection = validate_paper_analysis_inputs(rows, paper_policy)
    else:
        paths = trajectory_paths_for_analysis_tier(
            args.trajectory_root,
            args.analysis_tier,
        )
        records = load_trajectory_records(paths)
        records, paper_input_selection = select_paper_trajectory_records(
            records,
            paper_policy,
        )
        rows = build_activation_index(
            records,
            analysis_tier=args.analysis_tier,
            artifact_root=args.artifact_root,
            require_local=args.require_local,
            verify_checksums=args.verify_checksums,
        )
    write_activation_index(rows, args.output)
    summary = summarize_activation_index(rows)
    summary["paper_input_selection"] = paper_input_selection
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **summary,
                "index_path": args.output.as_posix(),
                "summary_path": args.summary_output.as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
