#!/usr/bin/env python3
"""Create final Scenario 1 diagnostic figures, tables, and a result manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.depth_degradation import load_prediction_jsonl  # noqa: E402
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    validate_paper_analysis_inputs,
)
from src.analysis.probe_analysis_figures import (  # noqa: E402
    REFERENCE_LAYER,
    figure_catalog,
    save_probe_analysis_figures,
)
from src.analysis.reporting_snapshot import (  # noqa: E402
    build_reporting_snapshot,
    load_reporting_snapshot,
    reporting_snapshot_analysis_tier,
    sha256_file,
    write_reporting_snapshot,
)
from src.extraction.saved_activations import (  # noqa: E402
    ACTIVATION_INDEX_ANALYSIS_TIERS,
    load_trajectory_records,
    trajectory_analysis_tier,
    trajectory_paths_for_analysis_tier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot saved probe scores without model inference or GPU compute."
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        default=PROJECT_ROOT
        / "results/scenario1/depth_analysis/depth_degradation.json",
    )
    parser.add_argument(
        "--all-layer-result",
        type=Path,
        default=PROJECT_ROOT
        / "results/scenario1/depth_analysis/all_layer_descriptive.json",
    )
    parser.add_argument(
        "--per-step-scores",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/per_step_probe_scores.jsonl",
    )
    parser.add_argument(
        "--snapshot-input",
        type=Path,
        help=(
            "Rebuild figures from a compact tracked reporting snapshot instead of "
            "local raw analysis outputs."
        ),
    )
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        default=(PROJECT_ROOT / "results/scenario1/reporting_snapshot.json"),
        help="Compact public snapshot written when raw analysis outputs are used.",
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/scenario1/trajectories/live",
        help="Live trajectory tree used to record model and decoding provenance.",
    )
    parser.add_argument(
        "--analysis-tier",
        choices=sorted(ACTIVATION_INDEX_ANALYSIS_TIERS),
        help=(
            "Required when rebuilding the snapshot from raw outputs. Selects one "
            "live trajectory namespace and must match the probe-score tier."
        ),
    )
    parser.add_argument(
        "--analysis-revision",
        help="Git revision that produced the raw analysis outputs (default: current HEAD).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/figures/paper",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/final_analysis",
    )
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    paper_policy = load_paper_input_policy(args.paper_input_policy)

    if args.snapshot_input:
        if args.analysis_revision:
            raise ValueError(
                "--analysis-revision cannot be combined with --snapshot-input."
            )
        snapshot = load_reporting_snapshot(args.snapshot_input)
        reference = snapshot["reference_result"]
        all_layer = snapshot["all_layer_result"]
        score_rows = snapshot["per_step_rows"]
        analysis_tier = reporting_snapshot_analysis_tier(snapshot)
        if args.analysis_tier and args.analysis_tier != analysis_tier:
            raise ValueError(
                "--analysis-tier does not match the reporting snapshot tier."
            )
        snapshot_path = args.snapshot_input
    else:
        if not args.analysis_tier:
            raise ValueError(
                "--analysis-tier is required when rebuilding from raw analysis outputs."
            )
        reference = _read_json(args.reference_result)
        all_layer = _read_json(args.all_layer_result)
        score_rows = load_prediction_jsonl(args.per_step_scores)
        source_artifacts = {
            "per_step_probe_scores": {
                "path": _relative(args.per_step_scores),
                "sha256": sha256_file(args.per_step_scores),
            },
            "reference_depth_result": {
                "path": _relative(args.reference_result),
                "sha256": sha256_file(args.reference_result),
            },
            "all_layer_descriptive_result": {
                "path": _relative(args.all_layer_result),
                "sha256": sha256_file(args.all_layer_result),
            },
            "paper_input_policy": {
                "path": _relative(args.paper_input_policy),
                "sha256": sha256_file(args.paper_input_policy),
            },
        }
        snapshot = build_reporting_snapshot(
            reference,
            all_layer,
            score_rows,
            analysis_revision=args.analysis_revision or _git_revision(),
            reporting_revision=_git_revision(),
            source_artifacts=source_artifacts,
            model_configs=_load_model_configs(
                args.trajectory_root,
                args.analysis_tier,
            ),
        )
        analysis_tier = reporting_snapshot_analysis_tier(snapshot)
        snapshot_path = args.snapshot_output

    paper_input_selection = validate_paper_analysis_inputs(
        score_rows,
        paper_policy,
    )
    snapshot["paper_input_selection"] = paper_input_selection
    if not args.snapshot_input:
        write_reporting_snapshot(snapshot, args.snapshot_output)

    figure_paths = save_probe_analysis_figures(
        reference,
        all_layer,
        score_rows,
        args.figure_dir,
        dpi=args.dpi,
    )

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    metric_table = args.analysis_dir / "reference_layer_metrics.csv"
    delta_table = args.analysis_dir / "reference_layer_depth_deltas.csv"
    _write_csv(_reference_metric_rows(reference), metric_table)
    _write_csv(_reference_delta_rows(reference), delta_table)

    sample = snapshot["sample"]
    catalog = figure_catalog()
    by_stem = {entry["stem"]: entry for entry in catalog}
    figure_entries = []
    for stem in sorted(by_stem):
        paths = [path for path in figure_paths if path.stem == stem]
        figure_entries.append(
            {
                **by_stem[stem],
                "files": [_relative(path) for path in sorted(paths)],
            }
        )

    manifest = {
        "schema_version": "spec_gap.final_analysis_manifest.v3",
        "analysis_tier": analysis_tier,
        "experiment_id": reference["experiment_id"],
        "analysis_revision": snapshot["analysis_revision"],
        "reporting_revision": snapshot["reporting_revision"],
        "model_configs": snapshot["model_configs"],
        "analysis_status": "construction_diagnostic_complete_behavioral_auroc_unavailable",
        "claim_scope": reference["claim_scope"],
        "label_target": "injection_present",
        "primary_analysis": {
            "thinking_mode": "off",
            "layer": REFERENCE_LAYER,
            "evaluation_method": "leave_one_match_group_out",
        },
        "sensitivity_analysis": {
            "thinking_mode": "on",
            "reason": "Thinking mode was added after the primary experiment was defined.",
        },
        "prespecified_layer_checks": {
            "reference": 40,
            "ablations": [32, 48],
            "all_layer_scan_role": "descriptive_only_no_post_hoc_selection",
        },
        "sample": sample,
        "source_artifacts": snapshot["source_artifacts"],
        "paper_input_selection": paper_input_selection,
        "paper_input_policy": {
            "path": _relative(args.paper_input_policy),
            "sha256": sha256_file(args.paper_input_policy),
        },
        "reporting_snapshot": {
            "path": _relative(snapshot_path),
            "sha256": _sha256(snapshot_path),
        },
        "tables": [_relative(metric_table), _relative(delta_table)],
        "figures": figure_entries,
        "supported_statements": [
            "The pipeline produced group-held-out construction-label scores for every saved agent step.",
            "Goldowsky-Dill, LAT, temporal path-mean classification, signed Temporal Divergence, calibration, and preliminary depth calculations ran on the controlled sample.",
            "Thinking-off is the primary analysis and thinking-on is reported separately as sensitivity analysis.",
            "All injected trajectories resisted and no unsafe executor action fired.",
        ],
        "unsupported_statements": [
            "The probes detect successful behavioral compromise.",
            "A final best activation layer has been identified.",
            "The two-group depth effect generalizes to new domains or scenarios.",
            "Domain effects can be separated from injection-wording effects in the two-group sample.",
            "Temporal path-mean aggregation or signed Temporal Divergence outperforms either baseline beyond this exploratory sample.",
        ],
    }
    manifest_path = args.analysis_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "analysis_manifest": _relative(manifest_path),
                "reporting_snapshot": _relative(snapshot_path),
                "figure_files": len(figure_paths),
                "figure_stems": [entry["stem"] for entry in catalog],
                "metric_table": _relative(metric_table),
                "delta_table": _relative(delta_table),
                "analysis_status": manifest["analysis_status"],
                "paper_input_policy_id": paper_input_selection["policy_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _reference_metric_rows(result: dict) -> list[dict]:
    rows = []
    for entry in result["depth_metrics"]:
        if int(entry["layer"]) != REFERENCE_LAYER:
            continue
        row = {
            "thinking_mode": entry["thinking_mode"],
            "probe_name": entry["probe_name"],
            "hop_mode": entry["hop_mode"],
            "layer": entry["layer"],
            "label_target": entry["label_target"],
            "n_match_groups": entry["n_match_groups"],
            "n_trajectories": entry["n_trajectories"],
        }
        for metric in (
            "auroc",
            "brier",
            "ece",
            "path_mean_auroc",
            "path_mean_brier",
            "path_mean_ece",
            "temporal_pre_anchor_mean",
            "temporal_post_anchor_mean",
            "temporal_divergence_mean",
        ):
            row[metric] = entry[metric]
            interval = entry["confidence_intervals"].get(metric)
            row[f"{metric}_ci_lower"] = interval["lower"] if interval else None
            row[f"{metric}_ci_upper"] = interval["upper"] if interval else None
        rows.append(row)
    return sorted(
        rows, key=lambda row: (row["thinking_mode"], row["probe_name"], row["hop_mode"])
    )


def _reference_delta_rows(result: dict) -> list[dict]:
    rows = []
    for entry in result["depth_comparisons"]:
        if int(entry["layer"]) != REFERENCE_LAYER:
            continue
        row = {
            "thinking_mode": entry["thinking_mode"],
            "probe_name": entry["probe_name"],
            "layer": entry["layer"],
            "comparison": entry["comparison"],
            "label_target": entry["label_target"],
            "n_match_groups": entry["n_match_groups"],
        }
        for metric, value in entry["deltas"].items():
            row[f"{metric}_delta"] = value
            interval = entry["confidence_intervals"].get(metric)
            row[f"{metric}_delta_ci_lower"] = interval["lower"] if interval else None
            row[f"{metric}_delta_ci_upper"] = interval["upper"] if interval else None
        rows.append(row)
    return sorted(rows, key=lambda row: (row["thinking_mode"], row["probe_name"]))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty result table to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load analysis result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Analysis result {path} must contain one object.")
    return payload


def _load_model_configs(trajectory_root: Path, analysis_tier: str) -> list[dict]:
    """Read one consistent model configuration for each thinking mode."""

    configs: dict[str, dict] = {}
    paths = trajectory_paths_for_analysis_tier(trajectory_root, analysis_tier)
    for payload in load_trajectory_records(paths):
        observed_tier = trajectory_analysis_tier(payload)
        if observed_tier != analysis_tier:
            raise ValueError(
                f"Live trajectory tier {observed_tier!r} does not match "
                f"{analysis_tier!r}."
            )
        model = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model, dict):
            continue
        mode = model.get("thinking_mode")
        if mode not in {"off", "on"}:
            continue
        normalized = {
            "analysis_tier": observed_tier,
            **{
                field: model.get(field)
                for field in (
                    "model_name",
                    "provider",
                    "model_revision",
                    "tokenizer_name",
                    "tokenizer_revision",
                    "dtype",
                    "seed",
                    "num_hidden_layers",
                    "thinking_mode",
                    "decoding_settings",
                )
            },
        }
        previous = configs.setdefault(str(mode), normalized)
        if previous != normalized:
            raise ValueError(
                f"Live trajectories contain inconsistent model settings for thinking {mode}."
            )
    if set(configs) != {"off", "on"}:
        raise ValueError(
            "Live trajectory provenance must cover thinking off and thinking on."
        )
    return [configs[mode] for mode in ("off", "on")]


def _git_revision() -> str:
    """Return the exact committed code revision used for snapshot generation."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise ValueError("Could not determine the current Git revision.")
    return revision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


if __name__ == "__main__":
    main()
