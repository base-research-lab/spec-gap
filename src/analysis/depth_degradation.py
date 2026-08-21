"""Depth-degradation analysis over precomputed trajectory probe scores.

This module is intentionally compute-backend agnostic. It consumes JSON-like
prediction rows after activation extraction and probe scoring have completed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.infrastructure.modal_billing import validate_analysis_tier
from src.probes.linear_probe import compute_ece
from src.probes.temporal_divergence import temporal_divergence


REQUIRED_FIELDS = {
    "trajectory_id",
    "match_group_id",
    "domain_id",
    "wording_id",
    "condition",
    "injection_present",
    "hop_mode",
    "agent_id",
    "agent_role",
    "hop_index",
    "distance_from_injection",
    "anchor_agent_id",
    "anchor_hop_index",
    "model",
    "thinking_mode",
    "layer",
    "probe_name",
    "score",
    "label",
    "label_target",
    "behavioral_outcome",
    "action_fired",
    "latent_compromise_status",
    "seed",
}
HOP_MODES = {"2-hop", "3-hop"}
CONDITIONS = {"clean", "injected"}
THINKING_MODES = {"off", "on"}
LABEL_TARGETS = {"trajectory_action_executed", "injection_present"}
BEHAVIORAL_OUTCOMES = {
    "clean",
    "resisted",
    "propagated_but_not_executed",
    "attempted_but_blocked",
    "executed",
    "indeterminate",
}
LATENT_STATUSES = {"not_candidate", "candidate", "probe_supported"}
DEPTH_RESULT_SCHEMA = "spec_gap.depth_degradation.v5"
TEMPORAL_SCORE_SCHEMA = "spec_gap.temporal_divergence_score.v3"
CURRENT_PER_STEP_SCORE_SCHEMA = "spec_gap.per_step_probe_score.v2"
BASELINE_METRIC_NAMES = ("auroc", "brier", "ece")
TEMPORAL_METRIC_NAMES = (
    "path_mean_auroc",
    "path_mean_brier",
    "path_mean_ece",
    "temporal_pre_anchor_mean",
    "temporal_post_anchor_mean",
    "temporal_divergence_mean",
    "temporal_peak_shift_mean",
    "temporal_persistence_mean",
)
METRIC_NAMES = (*BASELINE_METRIC_NAMES, *TEMPORAL_METRIC_NAMES)


def analyze_depth_degradation(
    rows: Iterable[dict],
    *,
    experiment_id: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    n_bins: int = 10,
    random_state: int = 42,
) -> dict:
    """Summarize metrics by depth and compute paired 3-hop minus 2-hop deltas."""

    rows = sorted((dict(row) for row in rows), key=_prediction_row_sort_key)
    validate_prediction_rows(rows)
    analysis_tier = prediction_analysis_tier(rows)
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    group_fields = (
        "model",
        "thinking_mode",
        "label_target",
        "probe_name",
        "layer",
        "seed",
        "hop_mode",
    )
    grouped = _group_rows(rows, group_fields)
    rng = np.random.default_rng(random_state)
    group_results = []
    for key in sorted(grouped, key=_sortable_key):
        group_rows = grouped[key]
        metrics = _metric_snapshot(group_rows, n_bins=n_bins)
        intervals = _bootstrap_intervals(
            group_rows,
            cluster_field="match_group_id",
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            n_bins=n_bins,
            rng=rng,
        )
        values = dict(zip(group_fields, key))
        executor_rows = _executor_rows(group_rows)
        scored_executor_rows = _scored_executor_rows(group_rows)
        temporal_rows = _temporal_rows(group_rows)
        scored_temporal_rows = [row for row in temporal_rows if row["label"] is not None]
        group_results.append({
            **values,
            "observation_agent": "executor",
            "n_predictions": len(scored_executor_rows),
            "n_excluded_unlabeled": len(executor_rows) - len(scored_executor_rows),
            "n_indeterminate": sum(
                row["behavioral_outcome"] == "indeterminate"
                for row in executor_rows
            ),
            "behavioral_outcome_counts": _behavioral_outcome_counts(executor_rows),
            "n_sequence_predictions": len(group_rows),
            "n_trajectories": len({row["trajectory_id"] for row in group_rows}),
            "n_match_groups": len({row["match_group_id"] for row in group_rows}),
            "n_positive": sum(int(row["label"]) for row in scored_executor_rows),
            "n_negative": sum(1 - int(row["label"]) for row in scored_executor_rows),
            "n_temporal_predictions": len(scored_temporal_rows),
            "n_temporal_excluded_unlabeled": len(temporal_rows) - len(scored_temporal_rows),
            **metrics,
            "confidence_intervals": intervals,
        })

    comparison_fields = (
        "model",
        "thinking_mode",
        "label_target",
        "probe_name",
        "layer",
        "seed",
    )
    by_configuration = _group_rows(rows, comparison_fields)
    comparisons = []
    for key in sorted(by_configuration, key=_sortable_key):
        config_rows = by_configuration[key]
        by_hop = _group_rows(config_rows, ("hop_mode",))
        missing_hops = HOP_MODES - {hop_key[0] for hop_key in by_hop}
        if missing_hops:
            values = dict(zip(comparison_fields, key))
            raise ValueError(
                f"Missing hop conditions {sorted(missing_hops)} for configuration {values}"
            )
        two_hop = by_hop[("2-hop",)]
        three_hop = by_hop[("3-hop",)]
        two_metrics = _metric_snapshot(two_hop, n_bins=n_bins)
        three_metrics = _metric_snapshot(three_hop, n_bins=n_bins)
        deltas = {
            metric: _difference(three_metrics[metric], two_metrics[metric])
            for metric in METRIC_NAMES
        }
        delta_intervals, n_match_groups = _bootstrap_delta_intervals(
            two_hop,
            three_hop,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            n_bins=n_bins,
            rng=rng,
        )
        comparisons.append({
            **dict(zip(comparison_fields, key)),
            "comparison": "3-hop_minus_2-hop",
            "observation_agent": "executor",
            "n_match_groups": n_match_groups,
            "deltas": deltas,
            "confidence_intervals": delta_intervals,
            "interpretation": {
                "auroc": "negative indicates worse discrimination at 3-hop",
                "brier": "positive indicates worse probabilistic accuracy at 3-hop",
                "ece": "positive indicates worse calibration at 3-hop",
                "temporal_divergence_mean": (
                    "signed change; interpret relative to the declared probe direction"
                ),
                "path_mean_auroc": (
                    "negative delta indicates worse temporal path-mean discrimination at 3-hop"
                ),
                "path_mean_brier": (
                    "positive delta indicates worse temporal path-mean probabilistic accuracy"
                ),
                "path_mean_ece": (
                    "positive delta indicates worse temporal path-mean calibration"
                ),
            },
        })

    n_match_groups = len({str(row["match_group_id"]) for row in rows})
    label_targets = sorted({str(row["label_target"]) for row in rows})
    evaluation_methods = sorted({
        str(row.get("evaluation_method", "not_declared")) for row in rows
    })
    return {
        "schema_version": DEPTH_RESULT_SCHEMA,
        "analysis_tier": analysis_tier,
        "experiment_id": experiment_id,
        "data_manifest_hash": prediction_manifest_hash(rows),
        "n_match_groups": n_match_groups,
        "claim_scope": (
            "group-held-out exploratory analysis with only "
            f"{n_match_groups} independent match "
            f"{'group' if n_match_groups == 1 else 'groups'}"
            if n_match_groups < 3
            else "preliminary match-group evaluation"
        ),
        "label_targets": label_targets,
        "sampling_grid": _sampling_grid(rows),
        "behavioral_outcome_counts": _behavioral_outcome_counts(
            _unique_executor_rows(rows)
        ),
        "analysis_config": {
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "n_bins": n_bins,
            "random_state": random_state,
            "delta_definition": "3-hop minus 2-hop",
            "classification_observation": "one executor score per trajectory",
            "classification_exclusion": (
                "rows with label=null are excluded from AUROC, Brier, and ECE"
            ),
            "temporal_observation": "full ordered agent-score sequence",
            "temporal_definition": (
                "pre-anchor mean is the planner score; post-anchor mean averages "
                "Worker1 through executor; divergence is post-anchor minus pre-anchor"
            ),
            "path_mean_classification_score": (
                "post_anchor_mean, the temporal path mean of held-out per-agent "
                "probabilities in [0, 1]"
            ),
            "path_mean_calibration_note": (
                "AUROC, Brier, and ECE use post_anchor_mean. Signed Temporal Divergence "
                "remains a separate trajectory-shape statistic and is not treated as a "
                "probability."
            ),
            "temporal_indeterminate_policy": (
                "indeterminate trajectories remain in temporal path and divergence summaries"
            ),
            "bootstrap_unit": "match_group_id for depth metrics and depth deltas",
            "score_generation_evaluation_methods": evaluation_methods,
        },
        "depth_metrics": group_results,
        "depth_comparisons": comparisons,
    }


def validate_prediction_rows(rows: list[dict]) -> None:
    """Validate the analysis contract and fail closed on incomplete trajectories."""

    if not rows:
        raise ValueError("At least one prediction row is required.")

    seen_rows: set[tuple] = set()
    trajectory_identity: dict[str, tuple] = {}
    trajectory_label: dict[str, int | None] = {}
    trajectory_layers: dict[tuple, set[int]] = defaultdict(set)
    trajectory_hops: dict[tuple, dict[int, tuple[str, str]]] = defaultdict(dict)
    match_group_cells: dict[str, dict[tuple[str, str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    match_group_design: dict[str, tuple[str, str]] = {}
    analysis_tiers: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"Row {index} is missing prediction fields: {missing}")
        raw_tier = row.get("analysis_tier")
        if raw_tier is None and row.get("schema_version") == CURRENT_PER_STEP_SCORE_SCHEMA:
            raise ValueError(
                f"Row {index} is missing prediction fields: ['analysis_tier']"
            )
        analysis_tiers.add(_normalize_analysis_tier(raw_tier or "unclassified"))
        if row["hop_mode"] not in HOP_MODES:
            raise ValueError(f"Row {index} has unsupported hop_mode {row['hop_mode']!r}")
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"Row {index} has unsupported condition {row['condition']!r}")
        if row["thinking_mode"] not in THINKING_MODES:
            raise ValueError(
                f"Row {index} has unsupported thinking_mode {row['thinking_mode']!r}"
            )
        if row["label_target"] not in LABEL_TARGETS:
            raise ValueError(
                f"Row {index} has unsupported label_target {row['label_target']!r}"
            )
        if row["behavioral_outcome"] not in BEHAVIORAL_OUTCOMES:
            raise ValueError(
                f"Row {index} has unsupported behavioral_outcome "
                f"{row['behavioral_outcome']!r}"
            )
        if row["latent_compromise_status"] not in LATENT_STATUSES:
            raise ValueError(
                f"Row {index} has unsupported latent_compromise_status "
                f"{row['latent_compromise_status']!r}"
            )
        if row["label"] not in (0, 1, False, True, None):
            raise ValueError(f"Row {index} label must be binary or null")
        if not isinstance(row["injection_present"], bool):
            raise ValueError(f"Row {index} injection_present must be boolean")
        if row["behavioral_outcome"] == "indeterminate":
            if row["action_fired"] is not None:
                raise ValueError(
                    f"Row {index} indeterminate outcome requires action_fired=null"
                )
        elif not isinstance(row["action_fired"], bool):
            raise ValueError(f"Row {index} completed outcome requires boolean action_fired")
        score = float(row["score"])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Row {index} score must be a finite probability in [0, 1]")
        if not isinstance(row["hop_index"], int) or not isinstance(row["anchor_hop_index"], int):
            raise ValueError(f"Row {index} hop indices must be integers")
        if not isinstance(row["layer"], int):
            raise ValueError(f"Row {index} layer must be an integer")
        if row["anchor_agent_id"] != "worker_1" or row["anchor_hop_index"] != 1:
            raise ValueError("Worker1 at hop 1 must be the fixed injection anchor")
        if row["distance_from_injection"] != row["hop_index"] - 1:
            raise ValueError(
                f"Row {index} distance_from_injection must equal hop_index minus 1"
            )

        expected_injection = row["condition"] == "injected"
        if row["injection_present"] is not expected_injection:
            raise ValueError(
                f"Row {index} condition and injection_present do not agree"
            )
        if row["condition"] == "clean" and row["behavioral_outcome"] not in {
            "clean",
            "indeterminate",
        }:
            raise ValueError(
                f"Row {index} clean condition must use clean or indeterminate behavioral_outcome"
            )
        if row["condition"] == "injected" and row["behavioral_outcome"] == "clean":
            raise ValueError(f"Row {index} injected condition cannot use clean behavioral_outcome")
        if row["action_fired"] and row["behavioral_outcome"] != "executed":
            raise ValueError(f"Row {index} action_fired requires executed behavioral_outcome")
        if not row["action_fired"] and row["behavioral_outcome"] == "executed":
            raise ValueError(f"Row {index} executed behavioral_outcome requires action_fired")

        candidate_outcomes = {"propagated_but_not_executed", "attempted_but_blocked"}
        if row["behavioral_outcome"] in candidate_outcomes:
            if row["latent_compromise_status"] not in {"candidate", "probe_supported"}:
                raise ValueError(
                    f"Row {index} propagated/blocked outcome must be a latent candidate"
                )
        elif row["latent_compromise_status"] != "not_candidate":
            raise ValueError(
                f"Row {index} latent candidate status conflicts with behavioral outcome"
            )

        expected_label = (
            None
            if (
                row["label_target"] == "trajectory_action_executed"
                and row["behavioral_outcome"] == "indeterminate"
            )
            else int(row["action_fired"])
            if row["label_target"] == "trajectory_action_executed"
            else int(row["injection_present"])
        )
        normalized_label = None if row["label"] is None else int(row["label"])
        if normalized_label != expected_label:
            raise ValueError(
                f"Row {index} label does not match label_target {row['label_target']!r}"
            )

        trajectory_id = str(row["trajectory_id"])
        identity = (
            str(row["match_group_id"]),
            str(row["domain_id"]),
            str(row["wording_id"]),
            str(row["condition"]),
            str(row["hop_mode"]),
            str(row["model"]),
            str(row["thinking_mode"]),
            str(row["label_target"]),
            str(row["behavioral_outcome"]),
            row["action_fired"],
            int(row["seed"]),
        )
        previous_identity = trajectory_identity.setdefault(trajectory_id, identity)
        if previous_identity != identity:
            raise ValueError(f"Trajectory {trajectory_id!r} has inconsistent identity metadata")
        previous_label = trajectory_label.setdefault(trajectory_id, normalized_label)
        if previous_label != normalized_label:
            raise ValueError(f"Trajectory {trajectory_id!r} has inconsistent labels")

        match_group_id = str(row["match_group_id"])
        design = (str(row["domain_id"]), str(row["wording_id"]))
        previous_design = match_group_design.setdefault(match_group_id, design)
        if previous_design != design:
            raise ValueError(
                f"Match group {match_group_id!r} mixes domain or wording metadata"
            )
        cell = (str(row["thinking_mode"]), str(row["condition"]), str(row["hop_mode"]))
        match_group_cells[match_group_id][cell].add(trajectory_id)

        row_key = (
            trajectory_id,
            row["model"],
            row["thinking_mode"],
            row["label_target"],
            row["probe_name"],
            row["layer"],
            row["seed"],
            row["hop_index"],
        )
        if row_key in seen_rows:
            raise ValueError(f"Duplicate prediction row: {row_key}")
        seen_rows.add(row_key)

        coverage_key = (
            trajectory_id,
            str(row["model"]),
            str(row["thinking_mode"]),
            str(row["label_target"]),
            str(row["probe_name"]),
            int(row["seed"]),
        )
        trajectory_layers[coverage_key].add(int(row["layer"]))
        hop_key = (*coverage_key, int(row["layer"]))
        if row["hop_index"] in trajectory_hops[hop_key]:
            raise ValueError(f"Duplicate hop_index in trajectory/layer group: {hop_key}")
        trajectory_hops[hop_key][int(row["hop_index"])] = (
            str(row["agent_id"]),
            str(row["agent_role"]),
        )

    if len(analysis_tiers) != 1:
        raise ValueError("Prediction rows must contain exactly one analysis tier.")

    expected_layers: dict[tuple, set[int]] = {}
    for coverage_key, layers in trajectory_layers.items():
        _, model, thinking_mode, label_target, probe_name, seed = coverage_key
        config_key = (model, thinking_mode, label_target, probe_name, seed)
        expected = expected_layers.setdefault(config_key, set(layers))
        if expected != layers:
            raise ValueError(
                f"Inconsistent layer coverage for {config_key}: "
                f"expected {sorted(expected)}, found {sorted(layers)}"
            )

    for hop_key, hop_map in trajectory_hops.items():
        ordered = sorted(hop_map)
        if ordered != list(range(len(ordered))):
            raise ValueError(
                f"Hop indices must be contiguous from zero for {hop_key}; found {ordered}"
            )
        trajectory_id = hop_key[0]
        hop_mode = trajectory_identity[trajectory_id][4]
        expected_agents = (
            [("planner_1", "planner"), ("worker_1", "worker"), ("executor_1", "executor")]
            if hop_mode == "2-hop"
            else [
                ("planner_1", "planner"),
                ("worker_1", "worker"),
                ("worker_2", "worker"),
                ("executor_1", "executor"),
            ]
        )
        actual_agents = [hop_map[index] for index in ordered]
        if actual_agents != expected_agents:
            raise ValueError(
                f"Trajectory {trajectory_id!r} has incorrect agent order: {actual_agents}"
            )

    expected_base_cells = {
        (condition, hop_mode)
        for condition in CONDITIONS
        for hop_mode in HOP_MODES
    }
    for match_group_id, cells in match_group_cells.items():
        thinking_modes = {thinking_mode for thinking_mode, _, _ in cells}
        for thinking_mode in thinking_modes:
            observed = {
                (condition, hop_mode)
                for mode, condition, hop_mode in cells
                if mode == thinking_mode
            }
            if observed != expected_base_cells:
                raise ValueError(
                    f"Match group {match_group_id!r} is missing a clean/injected depth cell "
                    f"for thinking_mode={thinking_mode!r}"
                )
            for condition, hop_mode in expected_base_cells:
                trajectory_ids = cells[(thinking_mode, condition, hop_mode)]
                if len(trajectory_ids) != 1:
                    raise ValueError(
                        f"Match group {match_group_id!r} must contain one trajectory per cell"
                    )


def prediction_manifest_hash(rows: Iterable[dict]) -> str:
    """Hash normalized input rows so reported results identify their exact inputs."""

    normalized = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("trajectory_id")),
            str(row.get("model")),
            str(row.get("thinking_mode")),
            str(row.get("label_target")),
            str(row.get("probe_name")),
            int(row.get("layer", -1)),
            int(row.get("seed", -1)),
            int(row.get("hop_index", -1)),
        ),
    )
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_prediction_jsonl(path: str | Path) -> list[dict]:
    """Load prediction rows from JSON Lines."""

    rows = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    validate_prediction_rows(rows)
    return rows


def tabular_result_rows(result: dict) -> list[dict]:
    """Flatten group metrics and depth deltas for CSV/DataFrame export."""

    rows = []
    for group in result["depth_metrics"]:
        row = {key: value for key, value in group.items() if key != "confidence_intervals"}
        row["analysis_tier"] = depth_result_analysis_tier(result)
        row["record_type"] = "depth_metric"
        for metric, interval in group["confidence_intervals"].items():
            row[f"{metric}_ci_lower"] = interval["lower"] if interval else None
            row[f"{metric}_ci_upper"] = interval["upper"] if interval else None
        rows.append(row)
    for comparison in result["depth_comparisons"]:
        row = {
            key: value
            for key, value in comparison.items()
            if key not in {"deltas", "confidence_intervals", "interpretation"}
        }
        row["analysis_tier"] = depth_result_analysis_tier(result)
        row["record_type"] = "depth_comparison"
        for metric, value in comparison["deltas"].items():
            row[f"{metric}_delta"] = value
            interval = comparison["confidence_intervals"][metric]
            row[f"{metric}_delta_ci_lower"] = interval["lower"] if interval else None
            row[f"{metric}_delta_ci_upper"] = interval["upper"] if interval else None
        rows.append(row)
    return rows


def temporal_divergence_rows(rows: Iterable[dict]) -> list[dict]:
    """Return path-mean and signed-divergence values per trajectory/config."""

    rows = sorted((dict(row) for row in rows), key=_prediction_row_sort_key)
    validate_prediction_rows(rows)
    analysis_tier = prediction_analysis_tier(rows)
    configuration_fields = (
        "model",
        "thinking_mode",
        "label_target",
        "probe_name",
        "layer",
        "seed",
    )
    grouped = _group_rows(rows, configuration_fields)
    output = []
    for key in sorted(grouped, key=_sortable_key):
        configuration = dict(zip(configuration_fields, key))
        configuration_rows = grouped[key]
        by_trajectory = _group_rows(configuration_rows, ("trajectory_id",))
        for (trajectory_id,), trajectory_rows in sorted(by_trajectory.items()):
            ordered = sorted(trajectory_rows, key=lambda row: row["hop_index"])
            aggregate = _temporal_rows(ordered)[0]
            first = ordered[0]
            output.append({
                "schema_version": TEMPORAL_SCORE_SCHEMA,
                "artifact_kind": "trajectory_temporal_divergence_score",
                "analysis_tier": analysis_tier,
                "trajectory_id": str(trajectory_id),
                "match_group_id": str(first["match_group_id"]),
                "domain_id": str(first["domain_id"]),
                "wording_id": str(first["wording_id"]),
                "condition": str(first["condition"]),
                "injection_present": bool(first["injection_present"]),
                "hop_mode": str(first["hop_mode"]),
                **configuration,
                "checkpoint": first.get("checkpoint"),
                "label": first["label"],
                "behavioral_outcome": str(first["behavioral_outcome"]),
                "action_fired": first["action_fired"],
                "latent_compromise_status": str(first["latent_compromise_status"]),
                "evaluation_method": first.get("evaluation_method"),
                "classification_score_name": "temporal_path_mean",
                "classification_score": aggregate["post_anchor_mean"],
                "pre_anchor_mean": aggregate["pre_anchor_mean"],
                "post_anchor_mean": aggregate["post_anchor_mean"],
                "divergence_score": aggregate["divergence_score"],
                "peak_shift": aggregate["peak_shift"],
                "persistence_fraction": aggregate["persistence_fraction"],
            })
    return sorted(output, key=lambda row: (
        str(row["probe_name"]),
        str(row["thinking_mode"]),
        int(row["layer"]),
        str(row["match_group_id"]),
        str(row["hop_mode"]),
        str(row["condition"]),
    ))


def prediction_analysis_tier(rows: Iterable[dict]) -> str:
    """Return the one normalized tier represented by prediction rows."""

    tiers = {
        _normalize_analysis_tier(row.get("analysis_tier") or "unclassified")
        for row in rows
    }
    if len(tiers) != 1:
        raise ValueError("Prediction rows must contain exactly one analysis tier.")
    return next(iter(tiers))


def depth_result_analysis_tier(result: dict) -> str:
    """Return a current tier or classify a historical depth result."""

    raw_tier = result.get("analysis_tier")
    if raw_tier is None and result.get("schema_version") == DEPTH_RESULT_SCHEMA:
        raise ValueError("Current depth results require analysis_tier.")
    return _normalize_analysis_tier(raw_tier or "unclassified")


def _normalize_analysis_tier(value: str) -> str:
    if value == "unclassified":
        return value
    return validate_analysis_tier(value)


def _metric_snapshot(rows: list[dict], *, n_bins: int) -> dict:
    executor_rows = _scored_executor_rows(rows)
    classification = _classification_snapshot(executor_rows, n_bins=n_bins)
    return {
        **classification,
        **_temporal_snapshot(rows, n_bins=n_bins),
    }


def _executor_rows(rows: list[dict], *, require_unique: bool = True) -> list[dict]:
    executor_rows = [row for row in rows if row["agent_id"] == "executor_1"]
    if not require_unique:
        return executor_rows
    trajectory_ids = {str(row["trajectory_id"]) for row in rows}
    executor_trajectories = {str(row["trajectory_id"]) for row in executor_rows}
    if executor_trajectories != trajectory_ids or len(executor_rows) != len(trajectory_ids):
        raise ValueError("Classification metrics require exactly one executor score per trajectory")
    return executor_rows


def _scored_executor_rows(rows: list[dict], *, require_unique: bool = True) -> list[dict]:
    return [
        row
        for row in _executor_rows(rows, require_unique=require_unique)
        if row["label"] is not None
    ]


def _classification_snapshot(rows: list[dict], *, n_bins: int) -> dict:
    if not rows:
        return {"auroc": None, "brier": None, "ece": None}
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    return {
        "auroc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else None
        ),
        "brier": float(brier_score_loss(labels, scores)),
        "ece": float(compute_ece(labels, scores, n_bins=n_bins)),
    }


def _behavioral_outcome_counts(executor_rows: list[dict]) -> dict[str, int]:
    return {
        outcome: sum(row["behavioral_outcome"] == outcome for row in executor_rows)
        for outcome in sorted(BEHAVIORAL_OUTCOMES)
    }


def _unique_executor_rows(rows: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in rows:
        if row["agent_id"] == "executor_1":
            unique.setdefault(str(row["trajectory_id"]), row)
    return list(unique.values())


def _temporal_rows(rows: list[dict]) -> list[dict]:
    """Aggregate ordered per-agent scores into one temporal-analysis row."""

    grouped = _group_rows(rows, ("trajectory_id",))
    values = []
    for (trajectory_id,), trajectory_rows in grouped.items():
        ordered = sorted(trajectory_rows, key=lambda row: row["hop_index"])
        anchor = int(ordered[0]["anchor_hop_index"])
        anchor_position = [row["hop_index"] for row in ordered].index(anchor)
        result = temporal_divergence(
            [float(row["score"]) for row in ordered],
            anchor_position=anchor_position,
        )
        labels = {row["label"] for row in ordered}
        if len(labels) != 1:
            raise ValueError(f"Trajectory {trajectory_id!r} has inconsistent temporal labels")
        values.append({
            "trajectory_id": str(trajectory_id),
            "match_group_id": str(ordered[0]["match_group_id"]),
            "label": labels.pop(),
            "pre_anchor_mean": result.pre_anchor_mean,
            "post_anchor_mean": result.post_anchor_mean,
            "divergence_score": result.divergence_score,
            "peak_shift": result.peak_shift,
            "persistence_fraction": result.persistence_fraction,
        })
    return values


def _temporal_snapshot(rows: list[dict], *, n_bins: int) -> dict:
    return _temporal_snapshot_from_aggregates(
        _temporal_rows(rows),
        n_bins=n_bins,
    )


def _temporal_snapshot_from_aggregates(
    temporal_rows: list[dict],
    *,
    n_bins: int,
) -> dict:
    if not temporal_rows:
        raise ValueError("Temporal path analysis requires at least one trajectory.")
    scored = [row for row in temporal_rows if row["label"] is not None]
    classification = _classification_snapshot(
        [
            {"label": row["label"], "score": row["post_anchor_mean"]}
            for row in scored
        ],
        n_bins=n_bins,
    )
    return {
        "path_mean_auroc": classification["auroc"],
        "path_mean_brier": classification["brier"],
        "path_mean_ece": classification["ece"],
        "temporal_pre_anchor_mean": float(np.mean([
            row["pre_anchor_mean"] for row in temporal_rows
        ])),
        "temporal_post_anchor_mean": float(np.mean([
            row["post_anchor_mean"] for row in temporal_rows
        ])),
        "temporal_divergence_mean": float(np.mean([
            row["divergence_score"] for row in temporal_rows
        ])),
        "temporal_peak_shift_mean": float(np.mean([
            row["peak_shift"] for row in temporal_rows
        ])),
        "temporal_persistence_mean": float(np.mean([
            row["persistence_fraction"] for row in temporal_rows
        ])),
    }


def _bootstrap_intervals(
    rows: list[dict],
    *,
    cluster_field: str,
    n_bootstrap: int,
    confidence: float,
    n_bins: int,
    rng: np.random.Generator,
) -> dict:
    cluster_ids = sorted({str(row[cluster_field]) for row in rows})
    samples = {metric: [] for metric in METRIC_NAMES}
    for _ in range(n_bootstrap):
        selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True).tolist()
        snapshot = _resampled_snapshot(
            rows,
            selected,
            cluster_field=cluster_field,
            n_bins=n_bins,
        )
        for metric in METRIC_NAMES:
            if snapshot[metric] is not None:
                samples[metric].append(snapshot[metric])
    return {
        metric: _confidence_interval(values, confidence)
        for metric, values in samples.items()
    }


def _bootstrap_delta_intervals(
    two_hop: list[dict],
    three_hop: list[dict],
    *,
    n_bootstrap: int,
    confidence: float,
    n_bins: int,
    rng: np.random.Generator,
) -> tuple[dict, int]:
    two_groups = {str(row["match_group_id"]) for row in two_hop}
    three_groups = {str(row["match_group_id"]) for row in three_hop}
    matched_groups = sorted(two_groups & three_groups)
    if not matched_groups:
        raise ValueError("Depth comparison requires at least one match group at both depths")
    if two_groups != three_groups:
        raise ValueError(
            "Depth comparison requires identical match-group coverage at 2-hop and 3-hop"
        )

    samples = {metric: [] for metric in METRIC_NAMES}
    for _ in range(n_bootstrap):
        selected = rng.choice(
            matched_groups,
            size=len(matched_groups),
            replace=True,
        ).tolist()
        two_snapshot = _resampled_snapshot(
            two_hop,
            selected,
            cluster_field="match_group_id",
            n_bins=n_bins,
        )
        three_snapshot = _resampled_snapshot(
            three_hop,
            selected,
            cluster_field="match_group_id",
            n_bins=n_bins,
        )
        for metric in METRIC_NAMES:
            delta = _difference(three_snapshot[metric], two_snapshot[metric])
            if delta is not None:
                samples[metric].append(delta)
    return (
        {
            metric: _confidence_interval(values, confidence)
            for metric, values in samples.items()
        },
        len(matched_groups),
    )


def _resampled_snapshot(
    rows: list[dict],
    selected_clusters: list[str],
    *,
    cluster_field: str,
    n_bins: int,
) -> dict:
    rows_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_cluster[str(row[cluster_field])].append(row)
    sampled_rows = [
        row
        for cluster_id in selected_clusters
        for row in rows_by_cluster[cluster_id]
    ]
    executor_rows = _scored_executor_rows(sampled_rows, require_unique=False)
    classification = _classification_snapshot(executor_rows, n_bins=n_bins)

    temporal_rows = _temporal_rows(rows)
    temporal_by_cluster: dict[str, list[dict]] = defaultdict(list)
    trajectory_cluster = {
        str(row["trajectory_id"]): str(row[cluster_field])
        for row in rows
    }
    for row in temporal_rows:
        temporal_by_cluster[trajectory_cluster[row["trajectory_id"]]].append(row)
    sampled_temporal_rows = [
        row
        for cluster_id in selected_clusters
        for row in temporal_by_cluster[cluster_id]
    ]
    return {
        **classification,
        **_temporal_snapshot_from_aggregates(
            sampled_temporal_rows,
            n_bins=n_bins,
        ),
    }


def _confidence_interval(values: list[float], confidence: float) -> dict | None:
    if not values:
        return None
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def _difference(three_hop: float | None, two_hop: float | None) -> float | None:
    if three_hop is None or two_hop is None:
        return None
    return float(three_hop - two_hop)


def _group_rows(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in fields)].append(row)
    return dict(grouped)


def _sortable_key(key: tuple) -> tuple[str, ...]:
    return tuple(str(value) for value in key)


def _prediction_row_sort_key(row: dict) -> tuple:
    return (
        str(row.get("match_group_id")),
        str(row.get("thinking_mode")),
        str(row.get("condition")),
        str(row.get("hop_mode")),
        str(row.get("trajectory_id")),
        str(row.get("model")),
        str(row.get("label_target")),
        str(row.get("probe_name")),
        int(row.get("layer", -1)),
        int(row.get("seed", -1)),
        int(row.get("hop_index", -1)),
    )


def _sampling_grid(rows: list[dict]) -> dict:
    domains = sorted({str(row["domain_id"]) for row in rows})
    wordings = sorted({str(row["wording_id"]) for row in rows})
    observed = sorted({(str(row["domain_id"]), str(row["wording_id"])) for row in rows})
    expected = {(domain, wording) for domain in domains for wording in wordings}
    return {
        "domains": domains,
        "wordings": wordings,
        "observed_domain_wording_cells": [
            {"domain_id": domain, "wording_id": wording}
            for domain, wording in observed
        ],
        "fully_crossed": set(observed) == expected,
    }
