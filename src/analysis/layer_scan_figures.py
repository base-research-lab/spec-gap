"""Figures for exploratory Scenario 1 layer scans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CHECKPOINT_ORDER = (
    "last_input_token",
    "last_reasoning_token",
    "last_visible_answer_token",
)
AGENT_ORDER = ("planner_1", "worker_1", "worker_2", "executor_1")
CHECKPOINT_LABELS = {
    "last_input_token": "Last input token",
    "last_reasoning_token": "Last reasoning token",
    "last_visible_answer_token": "Last visible-answer token",
}
AGENT_LABELS = {
    "planner_1": "Planner (negative control)",
    "worker_1": "Worker 1 (poison entry)",
    "worker_2": "Worker 2",
    "executor_1": "Executor",
}
CHECKPOINT_COLORS = {
    "last_input_token": "#2166ac",
    "last_reasoning_token": "#7b3294",
    "last_visible_answer_token": "#b2182b",
}


def save_layer_scan_figures(
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    dpi: int = 180,
) -> list[Path]:
    """Save mode grids and a dedicated planner-control figure."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    modes = sorted({
        stratum["thinking_mode"]
        for stratum in result.get("strata", [])
        if stratum.get("status") == "completed"
    })
    for mode in modes:
        path = destination / f"construction_layer_scan_thinking_{mode}.png"
        figure = _mode_grid(result, mode)
        _save_figure(figure, path, result=result, dpi=dpi)
        paths.append(path)

    control_path = destination / "planner_negative_controls.png"
    control_figure = _planner_control_figure(result)
    _save_figure(control_figure, control_path, result=result, dpi=dpi)
    paths.append(control_path)
    return paths


def _mode_grid(result: dict[str, Any], mode: str) -> plt.Figure:
    checkpoints = [
        checkpoint
        for checkpoint in CHECKPOINT_ORDER
        if _find_stratum(result, mode, "planner_1", checkpoint) is not None
    ]
    figure, axes = plt.subplots(
        len(AGENT_ORDER),
        len(checkpoints),
        figsize=(5.2 * len(checkpoints), 3.0 * len(AGENT_ORDER)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_index, agent_id in enumerate(AGENT_ORDER):
        for column_index, checkpoint in enumerate(checkpoints):
            axis = axes[row_index][column_index]
            stratum = _find_stratum(result, mode, agent_id, checkpoint)
            control = _find_control(result, mode, checkpoint)
            _style_control_background(axis, control)
            axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
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
                layers, aurocs = _layer_series(stratum)
                axis.plot(
                    layers,
                    aurocs,
                    color=CHECKPOINT_COLORS[checkpoint],
                    linewidth=1.8,
                )
                axis.text(
                    0.02,
                    0.04,
                    f"n={stratum['sample_count']}; groups={stratum['match_group_count']}",
                    transform=axis.transAxes,
                    fontsize=8,
                )
            if row_index == 0:
                axis.set_title(CHECKPOINT_LABELS[checkpoint])
            if column_index == 0:
                axis.set_ylabel(f"{AGENT_LABELS[agent_id]}\nMean AUROC")
            if row_index == len(AGENT_ORDER) - 1:
                axis.set_xlabel("Layer")
            axis.set_ylim(-0.02, 1.02)
            axis.set_xlim(0, 63)
            _control_badge(axis, control)

    figure.suptitle(
        f"Exploratory construction-label scan — thinking {mode}",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.008,
        (
            "Two independent match groups • leave-one-match-group-out • "
            "not final performance • red panels failed the planner control"
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.965))
    return figure


def _planner_control_figure(result: dict[str, Any]) -> plt.Figure:
    modes = sorted({
        control["thinking_mode"]
        for control in result["pre_injection_negative_control"].get(
            "checkpoint_controls", []
        )
    })
    figure, axes = plt.subplots(
        1,
        len(modes),
        figsize=(7 * len(modes), 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, mode in enumerate(modes):
        axis = axes[0][index]
        axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="Chance")
        for checkpoint in CHECKPOINT_ORDER:
            stratum = _find_stratum(result, mode, "planner_1", checkpoint)
            if stratum is None or stratum.get("status") != "completed":
                continue
            control = _find_control(result, mode, checkpoint)
            layers, aurocs = _layer_series(stratum)
            status = "PASS" if control and control["status"] == "passed" else "FAIL"
            axis.plot(
                layers,
                aurocs,
                color=CHECKPOINT_COLORS[checkpoint],
                linewidth=2,
                label=f"{CHECKPOINT_LABELS[checkpoint]} — {status}",
            )
        axis.set_title(f"Thinking {mode}")
        axis.set_xlabel("Layer")
        axis.set_xlim(0, 63)
        axis.set_ylim(-0.02, 1.02)
        axis.legend(fontsize=8, loc="best")
    axes[0][0].set_ylabel("Planner mean AUROC")
    figure.suptitle(
        "Pre-injection planner negative controls",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        (
            "Clean and injected planner prompts are identical. Separation here is "
            "spurious and disqualifies that mode/checkpoint from exploratory ranking."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return figure


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


def _layer_series(stratum: dict[str, Any]) -> tuple[list[int], list[float]]:
    pairs = sorted(
        (
            int(layer),
            float(metrics["auroc_mean"]),
        )
        for layer, metrics in stratum["layer_results"].items()
    )
    return [layer for layer, _ in pairs], [auroc for _, auroc in pairs]


def _style_control_background(
    axis: plt.Axes, control: dict[str, Any] | None
) -> None:
    if control is None:
        axis.set_facecolor("#f4f4f4")
    elif control["status"] == "passed":
        axis.set_facecolor("#f1f8f4")
    else:
        axis.set_facecolor("#fff0f0")


def _control_badge(axis: plt.Axes, control: dict[str, Any] | None) -> None:
    if control is None:
        label, color = "control: N/A", "#666666"
    elif control["status"] == "passed":
        label, color = "control: PASS", "#137333"
    else:
        label, color = "control: FAIL", "#b3261e"
    axis.text(
        0.98,
        0.96,
        label,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=color,
        fontweight="bold",
    )


def _save_figure(
    figure: plt.Figure,
    path: Path,
    *,
    result: dict[str, Any],
    dpi: int,
) -> None:
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={
            "Title": "SPEC-GAP exploratory construction-label layer scan",
            "Author": "SPEC-GAP",
            "Description": result["claim_scope"],
        },
    )
    plt.close(figure)
