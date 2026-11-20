"""Compact paper-style figures for the exploratory Scenario 1 layer scan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.analysis.figure_export import save_reproducible_figure
from src.infrastructure.model_profiles import ACTIVE_PROFILE, scale_qwen3_layers


MAIN_AGENT_ORDER = ("planner_1", "worker_1", "executor_1")
ALL_AGENT_ORDER = ("planner_1", "worker_1", "worker_2", "executor_1")
AGENT_LABELS = {
    "planner_1": "Planner control",
    "worker_1": "Worker 1",
    "worker_2": "Worker 2",
    "executor_1": "Executor",
}
FORMATS = ("png", "svg", "pdf")
QUALIFIED_CONTROL_STATUSES = frozenset({"passed", "passed_strict_input_control"})


def save_paper_layer_scan_figures(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    dpi: int = 600,
    filename_prefix: str = "scenario1_all_domains_",
) -> list[Path]:
    """Save two main figures and one appendix heatmap in three formats."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = (
        (
            "planner_negative_control",
            _planner_control_figure(result),
        ),
        (
            "shared_input_auroc_by_layer",
            _shared_input_figure(result),
        ),
        (
            "qualified_last_input_auroc_heatmap",
            _appendix_heatmap(result),
        ),
    )
    paths = []
    for stem, figure in figures:
        for suffix in FORMATS:
            path = destination / f"{filename_prefix}{stem}.{suffix}"
            _save(figure, path, result=result, dpi=dpi)
            paths.append(path)
        plt.close(figure)
    return paths


def _planner_control_figure(result: dict[str, Any]) -> plt.Figure:
    modes = _modes(result)
    match_group_count = len(_match_groups(result))
    mode_styles = {
        "off": {"color": "#0072B2", "linestyle": "-"},
        "on": {"color": "#D55E00", "linestyle": "--"},
    }
    with plt.rc_context(_paper_style()):
        figure, axis = plt.subplots(figsize=(6.6, 3.4))
        axis.axhline(
            0.5,
            color="#B7BEC8",
            linestyle=(0, (3, 3)),
            linewidth=0.9,
            zorder=0,
        )
        for mode in modes:
            stratum = _find_stratum(
                result,
                mode,
                "planner_1",
                "last_input_token",
            )
            if stratum is None or stratum.get("status") != "completed":
                continue
            control = _find_control(result, mode, "last_input_token")
            status, qualified, _ = _control_state(control)
            if not qualified:
                continue
            layers, means = _mean_series(stratum)
            style = mode_styles.get(
                mode,
                {"color": "#333333", "linestyle": "-"},
            )
            axis.plot(
                layers,
                means,
                linewidth=2.0,
                label=f"Thinking {mode} ({status})",
                zorder=2,
                **style,
            )
        _standard_axis(axis)
        axis.set_xlabel("Model layer")
        axis.set_ylabel("Mean held-out AUROC")
        axis.grid(axis="y", color="#ECEFF3", linewidth=0.7, zorder=0)
        axis.legend(loc="upper right", frameon=False)
        axis.text(
            62.5,
            0.515,
            "Random baseline = 0.5",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )
        figure.text(
            0.12,
            0.95,
            "Pre-retrieval planner negative control",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        figure.text(
            0.12,
            0.89,
            (
                "Identical clean and injected inputs; "
                f"n={match_group_count} held-out domains"
            ),
            ha="left",
            fontsize=8,
            color="#4B5563",
        )
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.17, top=0.80)
    return figure


def _shared_input_figure(result: dict[str, Any]) -> plt.Figure:
    modes = _modes(result)
    match_group_count = len(_match_groups(result))
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(
            len(MAIN_AGENT_ORDER),
            len(modes),
            figsize=(7.2, 6.4),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        panel_index = 0
        for row, agent_id in enumerate(MAIN_AGENT_ORDER):
            for column, mode in enumerate(modes):
                axis = axes[row][column]
                axis.axhline(
                    0.5,
                    color="#B7BEC8",
                    linestyle=(0, (3, 3)),
                    linewidth=0.8,
                    zorder=0,
                )
                stratum = _find_stratum(
                    result,
                    mode,
                    agent_id,
                    "last_input_token",
                )
                if stratum is None or stratum.get("status") != "completed":
                    axis.text(
                        0.5,
                        0.5,
                        "No eligible samples",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    _plot_fold_iqr(axis, stratum)
                    layers, means = _mean_series(stratum)
                    axis.plot(
                        layers,
                        means,
                        color="#111827",
                        linewidth=1.8,
                        zorder=2,
                    )
                _standard_axis(axis)
                axis.grid(axis="y", color="#F0F2F5", linewidth=0.6, zorder=0)
                if row == 0:
                    axis.set_title(f"Thinking {mode}")
                if column == 0:
                    axis.set_ylabel(
                        AGENT_LABELS[agent_id],
                        rotation=0,
                        ha="right",
                        va="center",
                        fontweight="bold",
                        labelpad=30,
                    )
                if row == len(MAIN_AGENT_ORDER) - 1:
                    axis.set_xlabel("Layer")
                axis.text(
                    -0.12,
                    1.04,
                    f"({chr(ord('a') + panel_index)})",
                    transform=axis.transAxes,
                    fontweight="bold",
                )
                panel_index += 1
        legend_handles = [
            Patch(
                facecolor="#9CA3AF",
                edgecolor="none",
                alpha=0.28,
                label=f"Domain IQR (n={match_group_count})",
            ),
            Line2D([0], [0], color="#111827", linewidth=1.8, label="Mean"),
            Line2D(
                [0],
                [0],
                color="#B7BEC8",
                linestyle=(0, (3, 3)),
                label="Random baseline = 0.5",
            ),
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=3,
            frameon=False,
            fontsize=7.5,
        )
        figure.suptitle(
            "Clean versus injected signal at the shared input checkpoint",
            fontweight="bold",
            y=0.99,
        )
        figure.text(
            0.012,
            0.5,
            "Mean held-out AUROC",
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0.09, 0, 1, 0.89))
    return figure


def _appendix_heatmap(result: dict[str, Any]) -> plt.Figure:
    match_group_count = len(_match_groups(result))
    modes = _modes(result)
    mode_matrices = []
    for mode in modes:
        matrix = []
        for agent_id in ALL_AGENT_ORDER:
            stratum = _find_stratum(
                result,
                mode,
                agent_id,
                "last_input_token",
            )
            control = _find_control(result, mode, "last_input_token")
            _, qualified, _ = _control_state(control)
            if (
                stratum is None
                or stratum.get("status") != "completed"
                or not qualified
            ):
                raise ValueError(
                    f"Qualified last-input stratum missing for {mode}/{agent_id}."
                )
            layers, means = _mean_series(stratum)
            if layers != list(range(ACTIVE_PROFILE.layer_count)):
                raise ValueError(
                    "Paper heatmap requires complete layers 0 through "
                    f"{ACTIVE_PROFILE.layer_count - 1}."
                )
            matrix.append(means)
        mode_matrices.append(np.asarray(matrix, dtype=float))

    color_map = LinearSegmentedColormap.from_list(
        "spec_gap_auroc",
        ((0.0, "#0072B2"), (0.5, "#F7F7F7"), (1.0, "#D55E00")),
    )
    color_norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(
            len(modes),
            1,
            figsize=(7.4, 4.5),
            sharex=True,
            squeeze=False,
        )
        image = None
        for row, (mode, matrix) in enumerate(zip(modes, mode_matrices)):
            axis = axes[row][0]
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=color_norm,
            )
            axis.set_yticks(range(len(ALL_AGENT_ORDER)))
            axis.set_yticklabels(
                [AGENT_LABELS[agent] for agent in ALL_AGENT_ORDER],
                fontsize=7.5,
            )
            axis.set_title(
                f"Thinking {mode}",
                loc="left",
                fontsize=8.5,
                fontweight="bold",
                pad=5,
            )
            axis.tick_params(axis="y", length=0, pad=7)
            for spine in axis.spines.values():
                spine.set_color("#D1D5DB")
                spine.set_linewidth(0.7)
        axes[-1][0].set_xticks(scale_qwen3_layers((0, 8, 16, 24, 32, 40, 48, 56, 63)))
        axes[-1][0].set_xlabel("Model layer")
        color_axis = figure.add_axes((0.91, 0.18, 0.018, 0.58))
        if image is None:
            raise ValueError("No qualified heatmap data were available.")
        color_bar = figure.colorbar(image, cax=color_axis)
        color_bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        color_bar.set_ticklabels(
            ["0.00", "0.25", "0.50\nrandom", "0.75", "1.00"]
        )
        color_bar.set_label("Mean held-out AUROC")
        color_bar.ax.axhline(0.5, color="#4B5563", linewidth=0.8)
        figure.text(
            0.11,
            0.95,
            "Clean versus injected AUROC across model layers",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        figure.text(
            0.11,
            0.90,
            (
                "Qualified last-input checkpoints; "
                f"n={match_group_count} held-out domains"
            ),
            ha="left",
            fontsize=8,
            color="#4B5563",
        )
        figure.subplots_adjust(
            left=0.20,
            right=0.88,
            bottom=0.13,
            top=0.82,
            hspace=0.34,
        )
    return figure


def _plot_fold_iqr(axis: plt.Axes, stratum: dict[str, Any]) -> None:
    layers, folds = _fold_series(stratum)
    values = np.asarray(folds, dtype=float)
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    axis.fill_between(
        layers,
        lower,
        upper,
        color="#9CA3AF",
        alpha=0.28,
        linewidth=0,
        zorder=1,
    )


def _mean_series(stratum: dict[str, Any]) -> tuple[list[int], list[float]]:
    pairs = sorted(
        (int(layer), float(metrics["auroc_mean"]))
        for layer, metrics in stratum["layer_results"].items()
    )
    return [layer for layer, _ in pairs], [mean for _, mean in pairs]


def _fold_series(stratum: dict[str, Any]) -> tuple[list[int], list[list[float]]]:
    ordered = sorted(
        (int(layer), metrics)
        for layer, metrics in stratum["layer_results"].items()
    )
    fold_count = len(ordered[0][1]["auroc_per_fold"])
    folds = [
        [float(metrics["auroc_per_fold"][fold]) for _, metrics in ordered]
        for fold in range(fold_count)
    ]
    return [layer for layer, _ in ordered], folds


def _find_stratum(
    result: dict[str, Any],
    mode: str,
    agent_id: str,
    checkpoint: str,
) -> dict[str, Any] | None:
    return next((
        stratum
        for stratum in result.get("strata", [])
        if stratum.get("thinking_mode") == mode
        and stratum.get("agent_id") == agent_id
        and stratum.get("checkpoint") == checkpoint
    ), None)


def _find_control(
    result: dict[str, Any], mode: str, checkpoint: str
) -> dict[str, Any] | None:
    return next((
        control
        for control in result.get("pre_injection_negative_control", {}).get(
            "checkpoint_controls", []
        )
        if control.get("thinking_mode") == mode
        and control.get("checkpoint") == checkpoint
    ), None)


def _control_state(
    control: dict[str, Any] | None,
) -> tuple[str, bool, str]:
    """Return a display label, qualification flag, and blocked-row message."""

    if control is None:
        return "NOT AVAILABLE", False, "Planner control not available"
    status = str(control.get("status", ""))
    if status in QUALIFIED_CONTROL_STATUSES:
        return "PASS", True, ""
    if status == "stochastic_null_uncalibrated":
        return "UNCALIBRATED", False, "Stochastic null not calibrated"
    return "FAIL", False, "Blocked by planner control"


def _modes(result: dict[str, Any]) -> list[str]:
    return sorted({
        stratum["thinking_mode"]
        for stratum in result.get("strata", [])
        if stratum.get("status") == "completed"
    })


def _match_groups(result: dict[str, Any]) -> list[str]:
    completed = [
        stratum
        for stratum in result.get("strata", [])
        if stratum.get("status") == "completed"
    ]
    for stratum in completed:
        match_groups = stratum.get("match_groups")
        if match_groups:
            return [str(group) for group in match_groups]
    count = max(
        (int(stratum.get("match_group_count", 0)) for stratum in completed),
        default=0,
    )
    return [f"group {index}" for index in range(1, count + 1)]


def _standard_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 63)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks(scale_qwen3_layers((0, 16, 32, 48, 63)))
    axis.set_yticks(np.linspace(0, 1, 6))
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
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


def _save(
    figure: plt.Figure,
    path: Path,
    *,
    result: dict[str, Any],
    dpi: int,
) -> None:
    common = {
        "Title": "SPEC-GAP exploratory construction-label layer analysis",
    }
    if path.suffix == ".png":
        metadata = {
            **common,
            "Author": "SPEC-GAP",
            "Description": result["claim_scope"],
        }
    elif path.suffix == ".pdf":
        metadata = {
            **common,
            "Author": "SPEC-GAP",
            "Subject": result["claim_scope"],
            "Keywords": "SPEC-GAP, activation probes, exploratory",
        }
    else:
        metadata = {
            **common,
            "Creator": "SPEC-GAP",
            "Description": result["claim_scope"],
            "Keywords": ["SPEC-GAP", "activation probes", "exploratory"],
        }
    save_reproducible_figure(
        figure,
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata=metadata,
    )
