"""Compact paper-style figures for the exploratory Scenario 1 layer scan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


CHECKPOINT_ORDER = (
    "last_input_token",
    "last_reasoning_token",
    "last_visible_answer_token",
)
CHECKPOINT_LABELS = {
    "last_input_token": "Last input",
    "last_reasoning_token": "Last reasoning",
    "last_visible_answer_token": "Last visible answer",
}
CHECKPOINT_COLORS = {
    "last_input_token": "#0072B2",
    "last_reasoning_token": "#CC79A7",
    "last_visible_answer_token": "#D55E00",
}
FOLD_COLORS = ("#56B4E9", "#E69F00")
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
    dpi: int = 300,
) -> list[Path]:
    """Save two main figures and one appendix heatmap in three formats."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figures = (
        ("figure1_planner_negative_control", _planner_control_figure(result)),
        ("figure2_shared_input_comparison", _shared_input_figure(result)),
        ("appendix_full_layer_heatmap", _appendix_heatmap(result)),
    )
    paths = []
    for stem, figure in figures:
        for suffix in FORMATS:
            path = destination / f"{stem}.{suffix}"
            _save(figure, path, result=result, dpi=dpi)
            paths.append(path)
        plt.close(figure)
    return paths


def _planner_control_figure(result: dict[str, Any]) -> plt.Figure:
    modes = _modes(result)
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(
            1,
            len(modes),
            figsize=(7.2, 3.4),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for column, mode in enumerate(modes):
            axis = axes[0][column]
            axis.axhline(0.5, color="#555555", linestyle=":", linewidth=1)
            for checkpoint in CHECKPOINT_ORDER:
                stratum = _find_stratum(result, mode, "planner_1", checkpoint)
                if stratum is None or stratum.get("status") != "completed":
                    continue
                control = _find_control(result, mode, checkpoint)
                status, qualified, _ = _control_state(control)
                linestyle = "-" if qualified else "--"
                _plot_fold_curves(
                    axis,
                    stratum,
                    color=CHECKPOINT_COLORS[checkpoint],
                    alpha=0.18,
                )
                layers, means = _mean_series(stratum)
                axis.plot(
                    layers,
                    means,
                    color=CHECKPOINT_COLORS[checkpoint],
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{CHECKPOINT_LABELS[checkpoint]} ({status})",
                )
            _standard_axis(axis)
            axis.set_title(f"Thinking {mode}")
            axis.set_xlabel("Layer")
            axis.text(
                -0.12,
                1.04,
                f"({chr(ord('a') + column)})",
                transform=axis.transAxes,
                fontweight="bold",
            )
            axis.legend(fontsize=7, loc="best", frameon=False)
        axes[0][0].set_ylabel("Planner mean AUROC")
        figure.suptitle(
            "Planner negative control before document retrieval",
            fontweight="bold",
            y=0.99,
        )
        figure.text(
            0.5,
            0.005,
            (
                "Faint lines are held-out match groups; bold lines are their mean. "
                "n=2 groups. Dashed checkpoints are not qualified for layer selection."
            ),
            ha="center",
            fontsize=7.5,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    return figure


def _shared_input_figure(result: dict[str, Any]) -> plt.Figure:
    modes = _modes(result)
    with plt.rc_context(_paper_style()):
        figure, axes = plt.subplots(
            len(MAIN_AGENT_ORDER),
            len(modes),
            figsize=(7.2, 7.0),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        panel_index = 0
        for row, agent_id in enumerate(MAIN_AGENT_ORDER):
            for column, mode in enumerate(modes):
                axis = axes[row][column]
                axis.axhline(0.5, color="#555555", linestyle=":", linewidth=1)
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
                    _plot_named_folds(axis, stratum)
                    layers, means = _mean_series(stratum)
                    axis.plot(layers, means, color="#000000", linewidth=1.8)
                    axis.text(
                        0.02,
                        0.04,
                        f"n={stratum['sample_count']}; groups={stratum['match_group_count']}",
                        transform=axis.transAxes,
                        fontsize=7,
                    )
                _standard_axis(axis)
                if row == 0:
                    axis.set_title(f"Thinking {mode}")
                if column == 0:
                    axis.set_ylabel(f"{AGENT_LABELS[agent_id]}\nMean AUROC")
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
            Line2D([0], [0], color=FOLD_COLORS[0], linewidth=1, label="Held-out group 1"),
            Line2D([0], [0], color=FOLD_COLORS[1], linewidth=1, label="Held-out group 2"),
            Line2D([0], [0], color="#000000", linewidth=1.8, label="Mean"),
            Line2D([0], [0], color="#555555", linestyle=":", label="Chance"),
        ]
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=4,
            frameon=False,
            fontsize=7.5,
        )
        figure.suptitle(
            "Clean–injected discrimination at the shared input checkpoint",
            fontweight="bold",
            y=0.99,
        )
        figure.text(
            0.5,
            0.005,
            (
                "Leave-one-match-group-out construction-label analysis. "
                "Exploratory only; not behavioral-compromise performance."
            ),
            ha="center",
            fontsize=7.5,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.90))
    return figure


def _appendix_heatmap(result: dict[str, Any]) -> plt.Figure:
    ordered = []
    for mode in _modes(result):
        for checkpoint in CHECKPOINT_ORDER:
            for agent_id in ALL_AGENT_ORDER:
                stratum = _find_stratum(result, mode, agent_id, checkpoint)
                if stratum is not None and stratum.get("status") == "completed":
                    ordered.append(stratum)
    matrix = []
    labels = []
    unqualified_rows = []
    qualification_messages = []
    for stratum in ordered:
        control = _find_control(
            result,
            stratum["thinking_mode"],
            stratum["checkpoint"],
        )
        status, qualified, message = _control_state(control)
        layers, means = _mean_series(stratum)
        if layers != list(range(64)):
            raise ValueError("Paper heatmap requires complete layers 0 through 63.")
        matrix.append(means if qualified else [np.nan] * 64)
        unqualified_rows.append(not qualified)
        qualification_messages.append(message)
        labels.append(
            f"{stratum['thinking_mode']} | {AGENT_LABELS[stratum['agent_id']]} | "
            f"{CHECKPOINT_LABELS[stratum['checkpoint']]} [{status}]"
        )

    with plt.rc_context(_paper_style()):
        figure, axis = plt.subplots(figsize=(8.2, max(5.0, 0.29 * len(ordered))))
        color_map = plt.get_cmap("cividis").copy()
        color_map.set_bad("#c7c7c7")
        image = axis.imshow(
            np.asarray(matrix, dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_xticks(range(0, 64, 8))
        axis.set_xticklabels(range(0, 64, 8))
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=6.6)
        axis.set_xlabel("Layer")
        axis.set_title(
            "Appendix: full exploratory layer scan",
            fontweight="bold",
            pad=10,
        )
        for row, unqualified in enumerate(unqualified_rows):
            if unqualified:
                axis.text(
                    31.5,
                    row,
                    qualification_messages[row],
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#222222",
                )
        color_bar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
        color_bar.set_label("Mean AUROC")
        figure.text(
            0.5,
            0.005,
            (
                "Grey rows are not qualified by the matched planner control. "
                "n=2 independent groups; values are descriptive and not used for "
                "final layer selection."
            ),
            ha="center",
            fontsize=7.5,
        )
        figure.tight_layout(rect=(0, 0.035, 1, 1))
    return figure


def _plot_fold_curves(
    axis: plt.Axes,
    stratum: dict[str, Any],
    *,
    color: str,
    alpha: float,
) -> None:
    layers, folds = _fold_series(stratum)
    for fold_values in folds:
        axis.plot(layers, fold_values, color=color, linewidth=0.7, alpha=alpha)


def _plot_named_folds(axis: plt.Axes, stratum: dict[str, Any]) -> None:
    layers, folds = _fold_series(stratum)
    for index, fold_values in enumerate(folds):
        axis.plot(
            layers,
            fold_values,
            color=FOLD_COLORS[index % len(FOLD_COLORS)],
            linewidth=0.8,
            alpha=0.72,
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


def _standard_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 63)
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks(range(0, 64, 16))
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
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata=metadata,
    )
