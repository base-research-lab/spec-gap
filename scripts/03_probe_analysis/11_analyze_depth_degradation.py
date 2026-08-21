#!/usr/bin/env python3
"""Analyze depth degradation from saved per-step probe-score JSONL rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.depth_degradation import (  # noqa: E402
    analyze_depth_degradation,
    load_prediction_jsonl,
    tabular_result_rows,
    temporal_divergence_rows,
)
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    validate_paper_analysis_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--temporal-output-jsonl", type=Path)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer indices, or 'all' (default).",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
    )
    args = parser.parse_args()

    prediction_rows = load_prediction_jsonl(args.predictions)
    paper_policy = load_paper_input_policy(args.paper_input_policy)
    paper_input_selection = validate_paper_analysis_inputs(
        prediction_rows,
        paper_policy,
    )
    selected_layers = _parse_layers(args.layers)
    if selected_layers is not None:
        prediction_rows = [
            row for row in prediction_rows if int(row["layer"]) in selected_layers
        ]
        if not prediction_rows:
            raise ValueError(f"No prediction rows matched layers {sorted(selected_layers)}.")
    result = analyze_depth_degradation(
        prediction_rows,
        experiment_id=args.experiment_id,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        n_bins=args.n_bins,
        random_state=args.random_state,
    )
    result["paper_input_selection"] = paper_input_selection
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if args.output_csv:
        rows = tabular_result_rows(result)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    if args.temporal_output_jsonl:
        trajectory_rows = temporal_divergence_rows(prediction_rows)
        args.temporal_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.temporal_output_jsonl.open("w") as handle:
            for row in trajectory_rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    print(json.dumps({
        "schema_version": result["schema_version"],
        "analysis_tier": result["analysis_tier"],
        "experiment_id": result["experiment_id"],
        "label_targets": result["label_targets"],
        "n_match_groups": result["n_match_groups"],
        "depth_metric_rows": len(result["depth_metrics"]),
        "depth_comparison_rows": len(result["depth_comparisons"]),
        "output_json": args.output_json.as_posix(),
        "output_csv": args.output_csv.as_posix() if args.output_csv else None,
        "temporal_output_jsonl": (
            args.temporal_output_jsonl.as_posix()
            if args.temporal_output_jsonl else None
        ),
    }, indent=2, sort_keys=True))


def _parse_layers(value: str) -> set[int] | None:
    if value.strip().lower() == "all":
        return None
    layers = set()
    for item in value.split(","):
        try:
            layer = int(item.strip())
        except ValueError as error:
            raise ValueError(f"Invalid layer list {value!r}.") from error
        if layer < 0:
            raise ValueError("Layer indices must be non-negative.")
        layers.add(layer)
    if not layers:
        raise ValueError("At least one layer must be selected.")
    return layers


if __name__ == "__main__":
    main()
