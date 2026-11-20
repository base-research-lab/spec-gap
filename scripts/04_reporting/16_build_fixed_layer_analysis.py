#!/usr/bin/env python3
"""Build the paper-ready fixed-layer Scenario 1 analysis bundle."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.model_profiles import ACTIVE_PROFILE, scale_qwen3_layers  # noqa: E402
from src.analysis.figure_export import save_reproducible_figure  # noqa: E402
from src.analysis.depth_degradation import (  # noqa: E402
    depth_result_analysis_tier,
    load_prediction_jsonl,
    prediction_analysis_tier,
)
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    validate_embedded_paper_input_audit,
    validate_paper_analysis_inputs,
)
from src.extraction.saved_activations import (  # noqa: E402
    activation_index_analysis_tier,
    load_activation_index,
)


# Qwen3-32B layer choices, mapped to the same relative depths for other profiles.
DEFAULT_MAIN_LAYERS = scale_qwen3_layers((16, 32, 40, 48, 63))
DEFAULT_ROBUSTNESS_LAYERS = scale_qwen3_layers((0, 8, 16, 24, 32, 40, 48, 56, 63))
PRIMARY_LAYER = ACTIVE_PROFILE.reference_layer
AGENT_ORDER = ("planner_1", "worker_1", "worker_2", "executor_1")
AGENT_LABELS = {
    "planner_1": "Planner",
    "worker_1": "Worker 1",
    "worker_2": "Worker 2",
    "executor_1": "Executor",
}
AGENT_FILE_LABELS = {
    "planner_1": "planner",
    "worker_1": "worker1",
    "worker_2": "worker2",
    "executor_1": "executor",
}
PROBE_LABELS = {
    "goldowsky_dill_logistic": "Goldowsky-Dill",
    "lat_contrast_pair_pca": "LAT",
}
PROBE_COLORS = {
    "goldowsky_dill_logistic": "#0072B2",
    "lat_contrast_pair_pca": "#D55E00",
}
PROBE_LINESTYLES = {
    "goldowsky_dill_logistic": "-",
    "lat_contrast_pair_pca": "--",
}
PROBE_MARKERS = {
    "goldowsky_dill_logistic": "o",
    "lat_contrast_pair_pca": "s",
}


def _display_path(path: Path) -> str:
    """Prefer a repository-relative source path when the file is in-tree."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


DOMAIN_LABELS = {
    "aihc": "AIHC",
    "fin": "Finance",
    "neuro": "Neuro",
    "macro": "Macro",
    "convex": "Convex",
    "kg": "Knowledge Graphs",
    "petro": "Petroleum",
    "policy": "Policy",
    "telecom": "Telecom",
}
DOMAIN_FILE_LABELS = {
    "aihc": "aihc",
    "fin": "finance",
    "neuro": "neuro",
    "macro": "macro",
    "convex": "convex",
    "kg": "knowledge_graphs",
    "petro": "petroleum",
    "policy": "policy",
    "telecom": "telecom",
}
DOMAIN_ORDER = (
    "aihc",
    "fin",
    "neuro",
    "macro",
    "convex",
    "kg",
    "petro",
    "policy",
    "telecom",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-layer tables, figures, and a manifest from saved "
            "group-held-out probe scores. This is CPU-only."
        )
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument(
        "--activation-index",
        type=Path,
        required=True,
        help=(
            "Verified activation-index JSONL used to create a portable, "
            "human-readable tensor catalog."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="Figure root. Defaults to <output-dir>/figures.",
    )
    parser.add_argument("--stem", required=True)
    parser.add_argument(
        "--file-prefix",
        default="scenario1",
        help=(
            "Short paper-facing filename prefix. The full run identifier remains "
            "recorded inside the index and summary."
        ),
    )
    parser.add_argument(
        "--main-layers",
        default=",".join(map(str, DEFAULT_MAIN_LAYERS)),
    )
    parser.add_argument(
        "--robustness-layers",
        default=",".join(map(str, DEFAULT_ROBUSTNESS_LAYERS)),
    )
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
    )
    args = parser.parse_args()

    main_layers = _parse_layers(args.main_layers)
    robustness_layers = _parse_layers(args.robustness_layers)
    if not set(main_layers).issubset(robustness_layers):
        raise ValueError("Every main-table layer must be in the robustness grid.")
    if args.primary_layer not in main_layers:
        raise ValueError("The primary layer must appear in the main table.")
    file_prefix = args.file_prefix.strip().lower()
    if not file_prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in file_prefix
    ):
        raise ValueError(
            "--file-prefix must use lowercase letters, numbers, or underscores."
        )

    score_rows = load_prediction_jsonl(args.scores)
    activation_rows = load_activation_index(args.activation_index)
    depth_result = json.loads(args.depth_result.read_text(encoding="utf-8"))
    paper_policy = load_paper_input_policy(args.paper_input_policy)
    index_selection = validate_paper_analysis_inputs(activation_rows, paper_policy)
    score_selection = validate_paper_analysis_inputs(score_rows, paper_policy)
    depth_selection = validate_embedded_paper_input_audit(
        depth_result,
        paper_policy,
    )
    analysis_tiers = {
        activation_index_analysis_tier(activation_rows),
        prediction_analysis_tier(score_rows),
        depth_result_analysis_tier(depth_result),
    }
    if len(analysis_tiers) != 1:
        raise ValueError("Fixed-layer inputs must contain one matching analysis tier.")
    selection_hashes = {
        selection["selected_trajectory_sha256"]
        for selection in (index_selection, score_selection, depth_selection)
    }
    if len(selection_hashes) != 1:
        raise ValueError(
            "Fixed-layer inputs do not contain the same paper-input cohort."
        )
    analysis_tier = next(iter(analysis_tiers))
    metrics = summarize_agent_metrics(score_rows, robustness_layers)
    domain_metrics = summarize_domain_metrics(score_rows, robustness_layers)
    main_metrics = [row for row in metrics if row["layer"] in main_layers]

    figure_dir = args.figure_dir or args.output_dir / "figures"
    all_domain_table_dir = args.output_dir / "tables" / "all_domains"
    domain_table_dir = args.output_dir / "tables" / "by_domain"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_domain_table_dir.mkdir(parents=True, exist_ok=True)
    all_table = all_domain_table_dir / (
        f"{file_prefix}_all_domains_robustness_layer_metrics.csv"
    )
    main_table = all_domain_table_dir / (
        f"{file_prefix}_all_domains_main_layer_metrics.csv"
    )
    _write_csv(metrics, all_table)
    _write_csv(main_metrics, main_table)
    activation_catalog = build_activation_artifact_catalog(activation_rows)
    activation_catalog_path = all_domain_table_dir / (
        f"{file_prefix}_activation_artifact_catalog.csv"
    )
    _write_csv(activation_catalog, activation_catalog_path)

    all_domain_figure_paths = plot_all_domain_figures(
        metrics,
        robustness_layers=robustness_layers,
        primary_layer=args.primary_layer,
        output_dir=figure_dir / "all_domains",
        stem=file_prefix,
        dpi=args.dpi,
    )
    domain_figure_paths = plot_standalone_domain_figures(
        domain_metrics,
        robustness_layers=robustness_layers,
        primary_layer=args.primary_layer,
        output_dir=figure_dir / "by_domain",
        stem=file_prefix,
        dpi=args.dpi,
    )
    domain_table_paths = write_domain_tables(
        domain_metrics,
        output_dir=domain_table_dir,
        stem=file_prefix,
    )
    manifest_path = args.output_dir / f"{file_prefix}_analysis_manifest.json"
    manifest = {
        "schema_version": "spec_gap.fixed_layer_analysis.v2",
        "analysis_tier": analysis_tier,
        "paper_input_selection": index_selection,
        "run_id": args.stem,
        "analysis_date": "2026-08-06",
        "claim_scope": "construction_label_diagnostic",
        "label_target": "injection_present",
        "evaluation_method": "leave_one_match_group_out",
        "model": sorted({str(row["model"]) for row in score_rows}),
        "model_revision": sorted({str(row["model_revision"]) for row in score_rows}),
        "domains": [
            DOMAIN_FILE_LABELS[domain] for domain in _ordered_domains(score_rows)
        ],
        "trajectory_count": len({str(row["trajectory_id"]) for row in score_rows}),
        "match_group_count": len({str(row["match_group_id"]) for row in score_rows}),
        "score_row_count": len(score_rows),
        "primary_thinking_mode": "off",
        "sensitivity_thinking_mode": "on",
        "primary_layer": args.primary_layer,
        "main_layers": list(main_layers),
        "robustness_layers": list(robustness_layers),
        "probes": list(PROBE_LABELS.values()),
        "source_files": {
            "activation_index": _display_path(args.activation_index),
            "probe_scores": _display_path(args.scores),
            "depth_analysis": _display_path(args.depth_result),
            "paper_input_policy": _display_path(args.paper_input_policy),
        },
        "depth_experiment_id": depth_result.get("experiment_id"),
        "outputs": {
            "all_domain_tables": [
                all_table.relative_to(args.output_dir).as_posix(),
                main_table.relative_to(args.output_dir).as_posix(),
                activation_catalog_path.relative_to(args.output_dir).as_posix(),
            ],
            "domain_tables": [
                path.relative_to(args.output_dir).as_posix()
                for path in domain_table_paths
            ],
            "all_domain_figures": [
                path.relative_to(args.output_dir).as_posix()
                for path in all_domain_figure_paths
            ],
            "domain_figures": [
                path.relative_to(args.output_dir).as_posix()
                for path in domain_figure_paths
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "score_rows": len(score_rows),
                "agent_metric_rows": len(metrics),
                "main_metric_rows": len(main_metrics),
                "main_layers": main_layers,
                "robustness_layers": robustness_layers,
                "primary_layer": args.primary_layer,
                "all_grid_table": all_table.as_posix(),
                "main_table": main_table.as_posix(),
                "manifest": manifest_path.as_posix(),
                "all_domain_standalone_figures": len(all_domain_figure_paths),
                "domain_standalone_figures": len(domain_figure_paths),
                "domain_tables": len(domain_table_paths),
                "activation_artifacts": len(activation_catalog),
                "activation_catalog": activation_catalog_path.as_posix(),
            },
            indent=2,
        )
    )


def build_activation_artifact_catalog(rows: list[dict]) -> list[dict]:
    """Return one portable catalog row per immutable activation tensor file."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        path = str(row["remote_storage_path"])
        grouped[path].append(row)

    catalog = []
    stable_fields = (
        "domain_id",
        "match_group_id",
        "trajectory_id",
        "treatment",
        "delegation_depth",
        "thinking_mode",
        "step_index",
        "agent_id",
        "agent_role",
        "checksum_sha256",
    )
    for path, cohort in sorted(grouped.items()):
        first = cohort[0]
        for field in stable_fields:
            values = {str(row[field]) for row in cohort}
            if len(values) != 1:
                raise ValueError(
                    f"Activation file {path} has inconsistent {field}: {values}."
                )
        layer_sets = {tuple(int(layer) for layer in row["layers"]) for row in cohort}
        if len(layer_sets) != 1:
            raise ValueError(f"Activation file {path} has inconsistent layers.")
        layers = next(iter(layer_sets))
        catalog.append(
            {
                "domain": DOMAIN_LABELS.get(
                    str(first["domain_id"]),
                    str(first["domain_id"]).replace("_", " ").title(),
                ),
                "match_group_id": first["match_group_id"],
                "trajectory_id": first["trajectory_id"],
                "treatment": first["treatment"],
                "delegation_depth": first["delegation_depth"],
                "thinking_mode": first["thinking_mode"],
                "step_index": first["step_index"],
                "agent_id": first["agent_id"],
                "agent_role": first["agent_role"],
                "checkpoints": ";".join(
                    sorted({str(row["checkpoint"]) for row in cohort})
                ),
                "layer_count": len(layers),
                "activation_file": path,
                "sha256": first["checksum_sha256"],
            }
        )
    return catalog


def summarize_agent_metrics(rows: list[dict], layers: tuple[int, ...]) -> list[dict]:
    selected = [row for row in rows if int(row["layer"]) in layers]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in selected:
        key = (
            str(row["thinking_mode"]),
            str(row["agent_id"]),
            str(row["probe_name"]),
            int(row["layer"]),
        )
        grouped[key].append(row)

    output = []
    for (thinking, agent, probe, layer), cohort in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            AGENT_ORDER.index(item[0][1]),
            item[0][2],
            item[0][3],
        ),
    ):
        labels = [int(row["label"]) for row in cohort]
        scores = [float(row["score"]) for row in cohort]
        by_fold: dict[str, list[dict]] = defaultdict(list)
        for row in cohort:
            by_fold[str(row["held_out_match_group_id"])].append(row)
        fold_aurocs = []
        for fold_rows in by_fold.values():
            fold_aurocs.append(
                _auroc(
                    [int(row["label"]) for row in fold_rows],
                    [float(row["score"]) for row in fold_rows],
                )
            )
        output.append(
            {
                "thinking_mode": thinking,
                "agent_id": agent,
                "agent_label": AGENT_LABELS[agent],
                "probe_name": probe,
                "probe_label": PROBE_LABELS[probe],
                "layer": layer,
                "n_predictions": len(cohort),
                "n_trajectories": len({str(row["trajectory_id"]) for row in cohort}),
                "n_match_groups": len(by_fold),
                "pooled_auroc": _auroc(labels, scores),
                "mean_fold_auroc": statistics.fmean(fold_aurocs),
                "fold_auroc_sd": statistics.pstdev(fold_aurocs),
                "fold_auroc_min": min(fold_aurocs),
                "fold_auroc_max": max(fold_aurocs),
                "brier": statistics.fmean(
                    (score - label) ** 2 for score, label in zip(scores, labels)
                ),
                "ece_10_bin": _ece(labels, scores, n_bins=10),
            }
        )
    return output


def summarize_domain_metrics(rows: list[dict], layers: tuple[int, ...]) -> list[dict]:
    """Summarize each independently held-out domain without averaging folds."""
    selected = [row for row in rows if int(row["layer"]) in layers]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in selected:
        domain = str(row["domain_id"])
        held_out_group = str(row["held_out_match_group_id"])
        key = (
            domain,
            held_out_group,
            str(row["thinking_mode"]),
            str(row["agent_id"]),
            str(row["probe_name"]),
            int(row["layer"]),
        )
        grouped[key].append(row)

    output = []
    for (domain, held_out_group, thinking, agent, probe, layer), cohort in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            AGENT_ORDER.index(item[0][3]),
            item[0][4],
            item[0][5],
        ),
    ):
        labels = [int(row["label"]) for row in cohort]
        scores = [float(row["score"]) for row in cohort]
        output.append(
            {
                "domain_id": domain,
                "domain_label": DOMAIN_LABELS.get(
                    domain, domain.replace("_", " ").title()
                ),
                "held_out_match_group_id": held_out_group,
                "thinking_mode": thinking,
                "agent_id": agent,
                "agent_label": AGENT_LABELS[agent],
                "probe_name": probe,
                "probe_label": PROBE_LABELS[probe],
                "layer": layer,
                "n_predictions": len(cohort),
                "n_trajectories": len({str(row["trajectory_id"]) for row in cohort}),
                "n_clean": sum(label == 0 for label in labels),
                "n_injected": sum(label == 1 for label in labels),
                "held_out_fold_auroc": _auroc(labels, scores),
                "brier": statistics.fmean(
                    (score - label) ** 2 for score, label in zip(scores, labels)
                ),
                "ece_10_bin": _ece(labels, scores, n_bins=10),
            }
        )
    return output


def plot_all_domain_figures(
    rows: list[dict],
    *,
    robustness_layers: tuple[int, ...],
    primary_layer: int,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Create one uncluttered nine-domain plot per thinking mode and agent."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for mode in ("off", "on"):
        for agent in AGENT_ORDER:
            cohort = [
                row
                for row in rows
                if row["thinking_mode"] == mode and row["agent_id"] == agent
            ]
            file_stem = (
                f"{stem}_all_domains_{AGENT_FILE_LABELS[agent]}_"
                f"thinking_{mode}_auroc_by_layer"
            )
            paths.extend(
                _plot_standalone_lines(
                    cohort,
                    value_field="mean_fold_auroc",
                    robustness_layers=robustness_layers,
                    primary_layer=primary_layer,
                    title=f"{AGENT_LABELS[agent]}, thinking {mode}",
                    subtitle="Mean AUROC across nine held-out domains",
                    output_dir=output_dir / f"thinking_{mode}",
                    file_stem=file_stem,
                    formats=("png", "pdf", "svg"),
                    dpi=dpi,
                )
            )
    return paths


def plot_standalone_domain_figures(
    rows: list[dict],
    *,
    robustness_layers: tuple[int, ...],
    primary_layer: int,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    """Create one single-panel plot per domain, thinking mode, and agent."""
    paths: list[Path] = []
    domains = _ordered_domains(rows)
    for domain in domains:
        domain_label = DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())
        domain_file_label = DOMAIN_FILE_LABELS.get(domain, domain)
        for mode in ("off", "on"):
            for agent in AGENT_ORDER:
                cohort = [
                    row
                    for row in rows
                    if row["domain_id"] == domain
                    and row["thinking_mode"] == mode
                    and row["agent_id"] == agent
                ]
                file_stem = (
                    f"{stem}_{domain_file_label}_{AGENT_FILE_LABELS[agent]}_"
                    f"thinking_{mode}_auroc_by_layer"
                )
                paths.extend(
                    _plot_standalone_lines(
                        cohort,
                        value_field="held_out_fold_auroc",
                        robustness_layers=robustness_layers,
                        primary_layer=primary_layer,
                        title=(
                            f"{domain_label}: {AGENT_LABELS[agent]}, thinking {mode}"
                        ),
                        subtitle="AUROC in this independently held-out domain fold",
                        output_dir=(
                            output_dir / domain_file_label / f"thinking_{mode}"
                        ),
                        file_stem=file_stem,
                        formats=("png", "pdf", "svg"),
                        dpi=dpi,
                    )
                )
    return paths


def _plot_standalone_lines(
    rows: list[dict],
    *,
    value_field: str,
    robustness_layers: tuple[int, ...],
    primary_layer: int,
    title: str,
    subtitle: str,
    output_dir: Path,
    file_stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> list[Path]:
    if not rows:
        raise ValueError(f"No rows available for standalone plot {file_stem}.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_publication_style())
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    primary_values: list[str] = []
    for probe in PROBE_LABELS:
        series = sorted(
            (row for row in rows if row["probe_name"] == probe),
            key=lambda row: row["layer"],
        )
        layer_values = [int(row["layer"]) for row in series]
        metric_values = [float(row[value_field]) for row in series]
        axis.plot(
            layer_values,
            metric_values,
            marker=PROBE_MARKERS[probe],
            markersize=4.2,
            markeredgewidth=0.7,
            linestyle=PROBE_LINESTYLES[probe],
            linewidth=1.7,
            color=PROBE_COLORS[probe],
            label=PROBE_LABELS[probe],
            zorder=3,
        )
        primary_row = next(
            (row for row in series if row["layer"] == primary_layer), None
        )
        if primary_row is not None:
            primary_values.append(
                f"{PROBE_LABELS[probe]} = {primary_row[value_field]:.3f}"
            )

    axis.axhline(
        0.5,
        color="#B7BEC8",
        linestyle=(0, (3, 3)),
        linewidth=0.9,
        zorder=0,
    )
    axis.axvline(
        primary_layer,
        color="#009E73",
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )
    axis.set_ylim(0.0, 1.02)
    axis.set_xlim(min(robustness_layers) - 1, max(robustness_layers) + 1)
    axis.set_xticks(robustness_layers)
    axis.set_xlabel("Model layer")
    axis.set_ylabel(
        "Mean held-out AUROC"
        if value_field == "mean_fold_auroc"
        else "Held-out fold AUROC"
    )
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#ECEFF3", linewidth=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#374151")
    axis.spines["bottom"].set_color("#374151")
    axis.legend(
        loc="lower right",
        frameon=False,
    )
    fig.text(0.12, 0.95, title, ha="left", fontsize=11, weight="bold")
    fig.text(0.12, 0.89, subtitle, ha="left", fontsize=8, color="#4B5563")
    axis.text(
        0.02,
        0.96,
        f"Layer {primary_layer}: " + " | ".join(primary_values),
        transform=axis.transAxes,
        va="top",
        fontsize=7.4,
        color="#374151",
        bbox={
            "boxstyle": "square,pad=0.18",
            "facecolor": "white",
            "alpha": 0.86,
            "edgecolor": "none",
        },
    )
    axis.text(
        62.5,
        0.515,
        "Random = 0.5",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6B7280",
    )
    axis.text(
        primary_layer + 0.8,
        0.03,
        f"Layer {primary_layer}",
        rotation=90,
        va="bottom",
        fontsize=7,
        color="#007A5E",
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.80)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = output_dir / f"{file_stem}.{suffix}"
        metadata = {
            "Title": title,
            "Author": "SPEC-GAP",
        }
        save_reproducible_figure(
            fig,
            path,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _publication_style() -> dict[str, object]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def write_domain_tables(rows: list[dict], *, output_dir: Path, stem: str) -> list[Path]:
    """Write one plainly named table per held-out domain and thinking mode."""
    paths: list[Path] = []
    domains = _ordered_domains(rows)
    for domain in domains:
        domain_file_label = DOMAIN_FILE_LABELS.get(domain, domain)
        for mode in ("off", "on"):
            cohort = [
                row
                for row in rows
                if row["domain_id"] == domain and row["thinking_mode"] == mode
            ]
            cohort.sort(
                key=lambda row: (
                    AGENT_ORDER.index(row["agent_id"]),
                    row["probe_name"],
                    row["layer"],
                )
            )
            directory = output_dir / domain_file_label
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (
                f"{stem}_{domain_file_label}_thinking_{mode}_layer_metrics.csv"
            )
            _write_csv(cohort, path)
            paths.append(path)
    return paths


def _ordered_domains(rows: list[dict]) -> list[str]:
    present = {str(row["domain_id"]) for row in rows}
    known = [domain for domain in DOMAIN_ORDER if domain in present]
    unknown = sorted(present.difference(DOMAIN_ORDER))
    return known + unknown


def _auroc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes.")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def _ece(labels: list[int], scores: list[float], *, n_bins: int) -> float:
    total = len(labels)
    error = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        members = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bin_index == n_bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        accuracy = statistics.fmean(labels[index] for index in members)
        confidence = statistics.fmean(scores[index] for index in members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def _parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("Layer lists must contain non-negative integers.")
    return layers


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
