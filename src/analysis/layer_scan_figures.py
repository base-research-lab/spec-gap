"""Figures for exploratory Scenario 1 layer scans."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.infrastructure.model_profiles import ACTIVE_PROFILE  # noqa: E402


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


def save_activation_coverage_status_figure(
    index_rows: list[dict[str, Any]],
    result: dict[str, Any],
    output_dir: str | Path,
    *,
    as_of_date: str,
    dpi: int = 180,
) -> Path:
    """Save a dated coverage/status figure without replacing analysis figures."""

    parsed_date = date.fromisoformat(as_of_date)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    match_groups = sorted({
        str(row["match_group_id"])
        for row in index_rows
        if row.get("match_group_id")
    })
    group_label = match_groups[0].upper() if len(match_groups) == 1 else "SCENARIO 1"
    group_slug = (
        match_groups[0].lower().replace("_", "-")
        if len(match_groups) == 1
        else "scenario1"
    )
    path = destination / (
        f"{as_of_date}_{group_slug}_activation_coverage_and_"
        "layer_selection_status.png"
    )

    primary_rows = [
        row for row in index_rows if row.get("thinking_mode") == "off"
    ]
    trajectory_ids = {
        str(row["trajectory_id"])
        for row in primary_rows
        if row.get("trajectory_id")
    }
    model_turns = {
        (str(row["trajectory_id"]), int(row["step_index"]))
        for row in primary_rows
        if row.get("trajectory_id") is not None
        and row.get("step_index") is not None
    }
    layer_counts = {
        len(row["layers"])
        for row in primary_rows
        if isinstance(row.get("layers"), list)
    }
    layer_count = max(layer_counts, default=0)

    figure, (coverage_axis, status_axis) = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.8),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )
    _draw_trajectory_coverage(coverage_axis, primary_rows)
    _draw_layer_selection_status(status_axis, result)

    readable_date = (
        f"{parsed_date.strftime('%B')} {parsed_date.day}, {parsed_date.year}"
    )
    figure.suptitle(
        f"{group_label} activation coverage and layer-selection status",
        fontsize=18,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        f"Current as of {readable_date} · Qwen3-32B · primary thinking-off runs",
        ha="center",
        fontsize=11,
        color="#4a5568",
    )
    figure.text(
        0.28,
        0.11,
        (
            f"{len(trajectory_ids)} trajectories  •  {len(model_turns)} model turns  •  "
            f"{len(primary_rows)} indexed checkpoints  •  {layer_count} layers"
        ),
        ha="center",
        fontsize=10,
        color="#334155",
    )
    figure.text(
        0.5,
        0.025,
        (
            "Preliminary validation only. This figure reports extraction coverage "
            "and analysis readiness—not probe performance."
        ),
        ha="center",
        fontsize=10,
        color="#334155",
        fontstyle="italic",
    )
    figure.tight_layout(rect=(0.025, 0.14, 0.975, 0.87))
    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        metadata={
            "Title": f"{group_label} activation coverage and layer-selection status",
            "Author": "SPEC-GAP",
            "Description": (
                f"Dated Scenario 1 extraction coverage status as of {as_of_date}; "
                "not a probe-performance result."
            ),
        },
    )
    plt.close(figure)
    return path


def _draw_trajectory_coverage(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    treatments = ("clean", "injected")
    depths = ("2-hop", "3-hop")
    available = {
        (str(row.get("delegation_depth")), str(row.get("treatment")))
        for row in rows
        if row.get("local_available", True)
    }
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_xticks((0.5, 1.5), ("Clean", "Injected"))
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, labelsize=11, pad=8)
    axis.set_yticks((1.5, 0.5), depths)
    axis.tick_params(axis="y", length=0, labelsize=11, pad=8)
    for row_index, depth in enumerate(depths):
        y = 1 - row_index
        for column_index, treatment in enumerate(treatments):
            present = (depth, treatment) in available
            face_color = "#dcefe7" if present else "#f8dddd"
            edge_color = "#2d7d70" if present else "#b44b4b"
            axis.add_patch(
                Rectangle(
                    (column_index + 0.04, y + 0.07),
                    0.92,
                    0.86,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=1.6,
                )
            )
            axis.text(
                column_index + 0.5,
                y + 0.58,
                "✓" if present else "—",
                ha="center",
                va="center",
                fontsize=23,
                fontweight="bold",
                color=edge_color,
            )
            axis.text(
                column_index + 0.5,
                y + 0.32,
                "saved" if present else "not available",
                ha="center",
                va="center",
                fontsize=10,
                color="#334155",
            )
    axis.set_title(
        "Primary trajectory coverage",
        fontsize=13,
        fontweight="bold",
        pad=42,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal")


def _draw_layer_selection_status(
    axis: plt.Axes,
    result: dict[str, Any],
) -> None:
    match_group_count = len(result.get("match_groups", []))
    required_group_count = 3
    axis.barh(
        [0],
        [match_group_count],
        height=0.34,
        color="#2d7d70",
        edgecolor="#24675e",
    )
    axis.axvline(
        required_group_count,
        color="#b44b4b",
        linestyle="--",
        linewidth=1.8,
    )
    axis.set_xlim(0, required_group_count + 0.45)
    axis.set_ylim(-1.0, 0.8)
    axis.set_yticks([])
    axis.set_xticks(
        range(required_group_count + 1),
        [str(value) for value in range(required_group_count + 1)],
    )
    axis.set_xlabel("Independent match groups", fontsize=10)
    axis.set_title(
        "Readiness for layer selection",
        fontsize=13,
        fontweight="bold",
        pad=18,
    )
    axis.text(
        match_group_count / 2 if match_group_count else 0.08,
        0,
        str(match_group_count),
        ha="center",
        va="center",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        required_group_count,
        0.28,
        "minimum: 3",
        ha="center",
        va="bottom",
        color="#9f3e3e",
        fontsize=9,
        fontweight="bold",
    )
    axis.text(
        (required_group_count + 0.45) / 2,
        -0.55,
        "LAYER SELECTION DEFERRED",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#9f3e3e",
    )
    axis.text(
        (required_group_count + 0.45) / 2,
        -0.73,
        "Additional independent match groups are needed.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )
    axis.spines[["top", "right", "left"]].set_visible(False)


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
            axis.set_xlim(0, ACTIVE_PROFILE.layer_count - 1)
            _control_badge(axis, control)

    figure.suptitle(
        f"Exploratory construction-label scan — thinking {mode}",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.008,
        _figure_scope_note(result),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.965))
    return figure


def _figure_scope_note(result: dict[str, Any]) -> str:
    match_groups = result.get("match_groups", [])
    match_group_count = len(match_groups)
    if match_group_count < 3:
        group_name = (
            str(match_groups[0]).upper()
            if match_group_count == 1
            else "two-group" if match_group_count == 2 else "limited-sample"
        )
        return (
            f"Preliminary {group_name} validation only; layer selection deferred "
            "pending additional match groups."
        )
    return (
        f"{match_group_count} independent match groups • "
        "leave-one-match-group-out • "
        "not final performance • red panels failed the planner control"
    )


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
        axis.set_xlim(0, ACTIVE_PROFILE.layer_count - 1)
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
