"""Publication-style figures for Scenario 1 probe and depth analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.analysis.figure_export import save_reproducible_figure
from src.infrastructure.model_profiles import ACTIVE_PROFILE


FORMATS = ("png", "svg", "pdf")
REFERENCE_LAYER = ACTIVE_PROFILE.reference_layer
ABLATION_LAYERS = ACTIVE_PROFILE.ablation_layers
LAYER_COUNT = ACTIVE_PROFILE.layer_count
PROBE_ORDER = ("goldowsky_dill_logistic", "lat_contrast_pair_pca")
PROBE_LABELS = {
    "goldowsky_dill_logistic": "Goldowsky–Dill",
    "lat_contrast_pair_pca": "LAT",
}
PROBE_COLORS = {
    "goldowsky_dill_logistic": "#0072B2",
    "lat_contrast_pair_pca": "#D55E00",
}
AGGREGATION_LABELS = {
    "executor": "Executor",
    "path_mean": "Temporal path mean",
}
CONDITION_COLORS = {"clean": "#0072B2", "injected": "#D55E00"}
OUTCOME_ORDER = (
    "clean",
    "resisted",
    "propagated_but_not_executed",
    "attempted_but_blocked",
    "executed",
    "indeterminate",
)
OUTCOME_LABELS = {
    "clean": "Clean",
    "resisted": "Resisted",
    "propagated_but_not_executed": "Propagated\nnot executed",
    "attempted_but_blocked": "Attempted\nblocked",
    "executed": "Executed",
    "indeterminate": "Indeterminate",
}


def save_probe_analysis_figures(
    reference_result: dict[str, Any],
    all_layer_result: dict[str, Any],
    per_step_rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Save the primary, sensitivity, and appendix figures in three formats."""

    _validate_inputs(reference_result, all_layer_result, per_step_rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            "figure3_primary_probe_metrics",
            _probe_metrics_figure(reference_result, thinking_mode="off"),
        ),
        (
            "figure4_depth_degradation",
            _depth_delta_figure(reference_result),
        ),
        (
            "figure5_temporal_profiles",
            _temporal_profile_figure(per_step_rows),
        ),
        (
            "figure6_behavioral_outcomes",
            _behavioral_outcome_figure(per_step_rows),
        ),
        (
            "appendix_thinking_on_probe_metrics",
            _probe_metrics_figure(reference_result, thinking_mode="on"),
        ),
        (
            "appendix_all_layer_robustness",
            _all_layer_figure(all_layer_result),
        ),
    )

    paths = []
    for stem, figure in figures:
        for suffix in FORMATS:
            path = destination / f"{stem}.{suffix}"
            _save(
                figure,
                path,
                claim_scope=reference_result["claim_scope"],
                dpi=dpi,
            )
            paths.append(path)
        plt.close(figure)
    return paths


def figure_catalog() -> list[dict[str, Any]]:
    """Return stable captions and interpretation boundaries for saved figures."""

    return [
        {
            "stem": "figure3_primary_probe_metrics",
            "role": "primary",
            "title": f"Thinking-off probe metrics at prespecified layer {REFERENCE_LAYER}",
            "caption": (
                "Group-held-out construction-label AUROC, Brier score, and ECE for "
                "executor-only and temporal path-mean aggregation at 2-hop and 3-hop. "
                "Signed Temporal Divergence is reported separately. Intervals resample "
                "independent match groups; the pooled group count is reported on the "
                "figure itself."
            ),
        },
        {
            "stem": "figure4_depth_degradation",
            "role": "primary_with_sensitivity_panel",
            "title": "Preliminary 3-hop minus 2-hop AUROC change",
            "caption": (
                f"Depth change at prespecified layer {REFERENCE_LAYER}. Thinking-off is the primary "
                "analysis and thinking-on is a late-added sensitivity analysis. Positive "
                "values indicate higher construction-label AUROC at 3-hop."
            ),
        },
        {
            "stem": "figure5_temporal_profiles",
            "role": "primary",
            "title": "Per-agent probe-score trajectories",
            "caption": (
                f"Thinking-off, layer-{REFERENCE_LAYER} held-out probabilities across the agent chain. "
                "Thin lines are independent match groups and bold lines are their mean."
            ),
        },
        {
            "stem": "figure6_behavioral_outcomes",
            "role": "primary_limitation",
            "title": "Observed behavioral outcomes",
            "caption": (
                "Observed executor outcome counts for injected and clean runs. "
                "Behavioral-compromise probe metrics are a separate, not-yet-computed "
                "analysis; when no unsafe action was executed in the sample, that "
                "metric is undefined."
            ),
        },
        {
            "stem": "appendix_thinking_on_probe_metrics",
            "role": "sensitivity",
            "title": f"Thinking-on sensitivity metrics at layer {REFERENCE_LAYER}",
            "caption": (
                "The same group-held-out construction-label analysis for the late-added "
                "thinking-on condition. It is not pooled with the primary thinking-off run."
            ),
        },
        {
            "stem": "appendix_all_layer_robustness",
            "role": "exploratory_appendix",
            "title": "Descriptive all-layer AUROC",
            "caption": (
                "All 64 layers are shown without selecting a best layer. Vertical markers "
                "identify the prespecified reference and ablation layers. The pooled "
                "group count is reported on the figure itself."
            ),
        },
    ]


def _probe_metrics_figure(
    result: dict[str, Any],
    *,
    thinking_mode: str,
) -> plt.Figure:
    metric_specs = (
        ("auroc", "path_mean_auroc", "AUROC", 0.0, 1.0),
        ("brier", "path_mean_brier", "Brier score ↓", 0.0, 0.8),
        ("ece", "path_mean_ece", "ECE ↓", 0.0, 0.9),
    )
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.25), squeeze=False)
        for panel, (executor_metric, temporal_metric, label, lower, upper) in enumerate(
            metric_specs
        ):
            axis = axes[0][panel]
            for probe in PROBE_ORDER:
                for aggregation, metric, marker, linestyle in (
                    ("executor", executor_metric, "o", "-"),
                    ("path_mean", temporal_metric, "s", "--"),
                ):
                    points = [
                        _depth_metric(
                            result,
                            thinking_mode=thinking_mode,
                            probe_name=probe,
                            layer=REFERENCE_LAYER,
                            hop_mode=hop_mode,
                        )
                        for hop_mode in ("2-hop", "3-hop")
                    ]
                    values = [float(point[metric]) for point in points]
                    errors = [_interval_error(point, metric) for point in points]
                    axis.errorbar(
                        (0, 1),
                        values,
                        yerr=np.asarray(errors).T,
                        color=PROBE_COLORS[probe],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=1.35,
                        markersize=4.5,
                        capsize=2.5,
                        label=(
                            f"{PROBE_LABELS[probe]} · "
                            f"{AGGREGATION_LABELS[aggregation]}"
                        ),
                    )
            if executor_metric == "auroc":
                axis.axhline(0.5, color="#777777", linestyle=":", linewidth=1)
            if executor_metric == "brier":
                axis.axhline(0.25, color="#777777", linestyle=":", linewidth=1)
            axis.set_xticks((0, 1), ("2-hop", "3-hop"))
            axis.set_ylabel(label)
            axis.set_ylim(lower, upper)
            axis.grid(axis="y", color="#dddddd", linewidth=0.5)
            _clean_axis(axis)
            axis.text(
                -0.12,
                1.04,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                fontweight="bold",
            )
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        title_prefix = "Primary" if thinking_mode == "off" else "Sensitivity"
        figure.suptitle(
            f"{title_prefix} construction-label diagnostics · thinking {thinking_mode}",
            fontweight="bold",
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.82))
    return figure


def _depth_delta_figure(result: dict[str, Any]) -> plt.Figure:
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
        for panel, mode in enumerate(("off", "on")):
            axis = axes[panel]
            axis.axhline(0.0, color="#666666", linestyle=":", linewidth=1)
            for aggregation, metric, marker, offset in (
                ("executor", "auroc", "o", -0.09),
                ("path_mean", "path_mean_auroc", "s", 0.09),
            ):
                for probe_index, probe in enumerate(PROBE_ORDER):
                    comparison = _depth_comparison(
                        result,
                        thinking_mode=mode,
                        probe_name=probe,
                        layer=REFERENCE_LAYER,
                    )
                    value = float(comparison["deltas"][metric])
                    error = _comparison_interval_error(comparison, metric)
                    axis.errorbar(
                        probe_index + offset,
                        value,
                        yerr=np.asarray(error).reshape(2, 1),
                        color=PROBE_COLORS[probe],
                        marker=marker,
                        linestyle="none",
                        capsize=3,
                        markersize=5,
                    )
            axis.set_xticks(range(len(PROBE_ORDER)))
            axis.set_xticklabels([PROBE_LABELS[probe] for probe in PROBE_ORDER])
            axis.set_title(
                "Thinking off · primary" if mode == "off" else "Thinking on · sensitivity"
            )
            axis.set_ylim(-1.05, 1.05)
            axis.grid(axis="y", color="#dddddd", linewidth=0.5)
            _clean_axis(axis)
            axis.text(
                -0.12,
                1.04,
                f"({chr(ord('a') + panel)})",
                transform=axis.transAxes,
                fontweight="bold",
            )
        axes[0].set_ylabel("AUROC change (3-hop − 2-hop)")
        legend = [
            Line2D([0], [0], color="#444444", marker="o", linestyle="none", label="Executor"),
            Line2D(
                [0],
                [0],
                color="#444444",
                marker="s",
                linestyle="none",
                label="Temporal path mean",
            ),
        ]
        figure.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            f"Preliminary depth change at prespecified layer {REFERENCE_LAYER}",
            fontweight="bold",
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.82))
    return figure


def _temporal_profile_figure(rows: Sequence[dict[str, Any]]) -> plt.Figure:
    filtered = [
        row for row in rows
        if row["thinking_mode"] == "off"
        and int(row["layer"]) == REFERENCE_LAYER
        and row["label_target"] == "injection_present"
    ]
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharey=True)
        panel = 0
        for probe_index, probe in enumerate(PROBE_ORDER):
            probe_rows = [row for row in filtered if row["probe_name"] == probe]
            for depth_index, hop_mode in enumerate(("2-hop", "3-hop")):
                axis = axes[probe_index][depth_index]
                depth_rows = [row for row in probe_rows if row["hop_mode"] == hop_mode]
                for condition in ("clean", "injected"):
                    trajectories = _trajectory_series(depth_rows, condition=condition)
                    for _, values in trajectories:
                        axis.plot(
                            range(len(values)),
                            values,
                            color=CONDITION_COLORS[condition],
                            alpha=0.25,
                            linewidth=0.8,
                        )
                    mean = np.mean([values for _, values in trajectories], axis=0)
                    axis.plot(
                        range(len(mean)),
                        mean,
                        color=CONDITION_COLORS[condition],
                        linewidth=2.0,
                        marker="o",
                        markersize=3.5,
                        label=condition.capitalize(),
                    )
                agent_labels = (
                    ("Planner", "Worker 1", "Executor")
                    if hop_mode == "2-hop"
                    else ("Planner", "Worker 1", "Worker 2", "Executor")
                )
                axis.axvline(1, color="#666666", linestyle=":", linewidth=1)
                axis.set_xticks(range(len(agent_labels)), agent_labels, rotation=20)
                axis.set_ylim(-0.02, 1.02)
                axis.grid(axis="y", color="#dddddd", linewidth=0.5)
                _clean_axis(axis)
                if probe_index == 0:
                    axis.set_title(hop_mode)
                if depth_index == 0:
                    axis.set_ylabel(f"{PROBE_LABELS[probe]}\nHeld-out score")
                axis.text(
                    -0.12,
                    1.04,
                    f"({chr(ord('a') + panel)})",
                    transform=axis.transAxes,
                    fontweight="bold",
                )
                panel += 1
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=2,
            frameon=False,
        )
        figure.suptitle(
            f"Per-agent score profiles · thinking off · layer {REFERENCE_LAYER}",
            fontweight="bold",
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    return figure


def _behavioral_outcome_figure(rows: Sequence[dict[str, Any]]) -> plt.Figure:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["trajectory_id"]), row)
    trajectories = list(unique.values())
    with plt.rc_context(_paper_style()):
        figure, axis = plt.subplots(figsize=(7.2, 3.5))
        x = np.arange(len(OUTCOME_ORDER))
        width = 0.36
        overall_max = 0
        for mode_index, (mode, color) in enumerate((
            ("off", "#0072B2"),
            ("on", "#E69F00"),
        )):
            values = [
                sum(
                    row["thinking_mode"] == mode
                    and row["behavioral_outcome"] == outcome
                    for row in trajectories
                )
                for outcome in OUTCOME_ORDER
            ]
            bars = axis.bar(
                x + (mode_index - 0.5) * width,
                values,
                width,
                color=color,
                label=f"Thinking {mode}",
            )
            axis.bar_label(bars, padding=2, fontsize=7)
            overall_max = max(overall_max, max(values) if values else 0)
        axis.set_xticks(x, [OUTCOME_LABELS[outcome] for outcome in OUTCOME_ORDER])
        axis.set_ylabel("Trajectory count")
        axis.set_ylim(0, max(1.0, overall_max * 1.25 + 0.25))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        _clean_axis(axis)
        axis.legend(frameon=False, ncol=2, loc="upper right")
        axis.set_title("Observed executor outcomes", fontweight="bold")
        figure.tight_layout(rect=(0, 0.02, 1, 1))
    return figure


def _all_layer_figure(result: dict[str, Any]) -> plt.Figure:
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True, sharey=True)
        for mode_index, mode in enumerate(("off", "on")):
            for probe_index, probe in enumerate(PROBE_ORDER):
                axis = axes[mode_index][probe_index]
                for hop_mode, color in (("2-hop", "#0072B2"), ("3-hop", "#D55E00")):
                    points = sorted(
                        (
                            row for row in result["depth_metrics"]
                            if row["thinking_mode"] == mode
                            and row["probe_name"] == probe
                            and row["hop_mode"] == hop_mode
                        ),
                        key=lambda row: int(row["layer"]),
                    )
                    layers = [int(row["layer"]) for row in points]
                    axis.plot(
                        layers,
                        [row["auroc"] for row in points],
                        color=color,
                        linewidth=1.0,
                        label=f"{hop_mode} · executor",
                    )
                    axis.plot(
                        layers,
                        [row["path_mean_auroc"] for row in points],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        label=f"{hop_mode} · path mean",
                    )
                axis.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
                for layer, linestyle in (
                    (ABLATION_LAYERS[0], ":"),
                    (REFERENCE_LAYER, "-"),
                    (ABLATION_LAYERS[1], ":"),
                ):
                    axis.axvline(layer, color="#888888", linestyle=linestyle, linewidth=0.7)
                axis.set_xlim(0, 63)
                axis.set_ylim(-0.02, 1.02)
                axis.set_xticks(range(0, 64, 8))
                axis.grid(axis="y", color="#e5e5e5", linewidth=0.4)
                _clean_axis(axis)
                axis.set_title(
                    f"{PROBE_LABELS[probe]} · thinking {mode}"
                    + (" (primary)" if mode == "off" else " (sensitivity)")
                )
                if probe_index == 0:
                    axis.set_ylabel("Construction-label AUROC")
                if mode_index == 1:
                    axis.set_xlabel("Layer")
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=4,
            frameon=False,
        )
        figure.suptitle(
            "All-layer robustness without post-hoc layer selection",
            fontweight="bold",
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    return figure


def _trajectory_series(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
) -> list[tuple[str, list[float]]]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            by_trajectory[str(row["trajectory_id"])].append(row)
    series = []
    for trajectory_id, trajectory_rows in sorted(by_trajectory.items()):
        ordered = sorted(trajectory_rows, key=lambda row: int(row["hop_index"]))
        expected = list(range(len(ordered)))
        observed = [int(row["hop_index"]) for row in ordered]
        if observed != expected:
            raise ValueError(
                f"Trajectory {trajectory_id!r} has incomplete score positions {observed}."
            )
        series.append((trajectory_id, [float(row["score"]) for row in ordered]))
    if not series:
        raise ValueError(f"No {condition!r} trajectories are available for a profile panel.")
    return series


def _depth_metric(
    result: dict[str, Any],
    *,
    thinking_mode: str,
    probe_name: str,
    layer: int,
    hop_mode: str,
) -> dict[str, Any]:
    matches = [
        row for row in result["depth_metrics"]
        if row["thinking_mode"] == thinking_mode
        and row["probe_name"] == probe_name
        and int(row["layer"]) == layer
        and row["hop_mode"] == hop_mode
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one depth metric for "
            f"{thinking_mode}/{probe_name}/layer-{layer}/{hop_mode}."
        )
    return matches[0]


def _depth_comparison(
    result: dict[str, Any],
    *,
    thinking_mode: str,
    probe_name: str,
    layer: int,
) -> dict[str, Any]:
    matches = [
        row for row in result["depth_comparisons"]
        if row["thinking_mode"] == thinking_mode
        and row["probe_name"] == probe_name
        and int(row["layer"]) == layer
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one depth comparison for {thinking_mode}/{probe_name}/layer-{layer}."
        )
    return matches[0]


def _interval_error(row: dict[str, Any], metric: str) -> tuple[float, float]:
    value = float(row[metric])
    interval = row["confidence_intervals"].get(metric)
    if interval is None:
        return (0.0, 0.0)
    return (
        max(0.0, value - float(interval["lower"])),
        max(0.0, float(interval["upper"]) - value),
    )


def _comparison_interval_error(
    row: dict[str, Any], metric: str
) -> tuple[float, float]:
    value = float(row["deltas"][metric])
    interval = row["confidence_intervals"].get(metric)
    if interval is None:
        return (0.0, 0.0)
    return (
        max(0.0, value - float(interval["lower"])),
        max(0.0, float(interval["upper"]) - value),
    )


def _validate_inputs(
    reference_result: dict[str, Any],
    all_layer_result: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> None:
    for name, result in (
        ("reference", reference_result),
        ("all-layer", all_layer_result),
    ):
        if result.get("label_targets") != ["injection_present"]:
            raise ValueError(f"{name} result must be an injection-present diagnostic.")
        if int(result.get("n_match_groups", 0)) < 2:
            raise ValueError(
                f"{name} result must declare at least two independent match groups "
                "for a leave-one-match-group-out comparison."
            )
    reference_layers = {int(row["layer"]) for row in reference_result["depth_metrics"]}
    if not {REFERENCE_LAYER, *ABLATION_LAYERS}.issubset(reference_layers):
        raise ValueError(
            f"Reference result must contain layers {ABLATION_LAYERS[0]}, "
            f"{REFERENCE_LAYER}, and {ABLATION_LAYERS[1]}."
        )
    all_layers = {int(row["layer"]) for row in all_layer_result["depth_metrics"]}
    if all_layers != set(range(LAYER_COUNT)):
        raise ValueError(
            f"All-layer result must contain layers 0 through {LAYER_COUNT - 1}."
        )
    if not rows:
        raise ValueError("Per-step probe-score rows are required for profile figures.")


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _paper_style() -> dict[str, Any]:
    return {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 11,
        "font.family": "DejaVu Sans",
    }


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    claim_scope: str,
    dpi: int,
) -> None:
    common = {"Title": "SPEC-GAP Scenario 1 probe analysis"}
    if path.suffix == ".png":
        metadata = {
            **common,
            "Author": "SPEC-GAP",
            "Description": claim_scope,
        }
    elif path.suffix == ".pdf":
        metadata = {
            **common,
            "Author": "SPEC-GAP",
            "Subject": claim_scope,
            "Keywords": "SPEC-GAP, activation probes, Temporal Divergence",
        }
    else:
        metadata = {
            **common,
            "Creator": "SPEC-GAP",
            "Description": claim_scope,
            "Keywords": ["SPEC-GAP", "activation probes", "Temporal Divergence"],
        }
    save_reproducible_figure(
        figure,
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata=metadata,
    )
