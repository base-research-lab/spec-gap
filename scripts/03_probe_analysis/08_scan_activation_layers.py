#!/usr/bin/env python3
"""Run an exploratory all-layer scan over the saved Scenario 1 activations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.layer_scan import (  # noqa: E402
    layer_scan_table,
    run_construction_layer_scan,
)
from src.analysis.activation_controls import (  # noqa: E402
    activation_control_table,
    run_activation_control_audit,
)
from src.extraction.saved_activations import (  # noqa: E402
    LABEL_TARGETS,
    load_activation_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all saved layers with leave-one-match-group-out evaluation. "
            "This is CPU-only and does not start Modal compute."
        )
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/activation_index.jsonl",
    )
    parser.add_argument(
        "--label-target",
        choices=sorted(LABEL_TARGETS),
        default="injection_present",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/construction_layer_scan.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/construction_layer_scan.csv",
    )
    parser.add_argument(
        "--control-output-json",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/activation_control_audit.json",
    )
    parser.add_argument(
        "--control-output-csv",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/activation_control_pairs.csv",
    )
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    index_rows = load_activation_index(args.index)
    control_audit = run_activation_control_audit(
        index_rows,
        verify_checksums=not args.skip_checksums,
    )
    args.control_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.control_output_json.write_text(
        json.dumps(control_audit, indent=2, sort_keys=True) + "\n"
    )
    control_table = activation_control_table(control_audit)
    args.control_output_csv.parent.mkdir(parents=True, exist_ok=True)
    control_fieldnames = sorted({key for row in control_table for key in row})
    with args.control_output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=control_fieldnames)
        writer.writeheader()
        writer.writerows(control_table)

    result = run_construction_layer_scan(
        index_rows,
        label_target=args.label_target,
        control_audit=control_audit,
        verify_checksums=not args.skip_checksums,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    table = layer_scan_table(result)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in table for key in row})
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)

    print(json.dumps({
        "schema_version": result["schema_version"],
        "label_target": result["label_target"],
        "selection_status": result["selection_status"],
        "layer_selection_allowed": result["layer_selection_allowed"],
        "negative_control_status": result["pre_injection_negative_control"]["status"],
        "strict_planner_input_control": control_audit[
            "strict_planner_input_control"
        ]["status"],
        "qualified_mode_checkpoints": result["pre_injection_negative_control"].get(
            "qualified_mode_checkpoints", []
        ),
        "blocked_mode_checkpoints": result["pre_injection_negative_control"].get(
            "blocked_mode_checkpoints", []
        ),
        "completed_strata": result["completed_strata"],
        "skipped_strata": result["skipped_strata"],
        "output_json": args.output_json.as_posix(),
        "output_csv": args.output_csv.as_posix(),
        "control_output_json": args.control_output_json.as_posix(),
        "control_output_csv": args.control_output_csv.as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
