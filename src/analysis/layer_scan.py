"""Exploratory, match-group-safe layer scans over saved activations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from src.extraction.saved_activations import (
    LABEL_TARGETS,
    load_probe_activation_batch,
)
from src.probes.linear_probe import evaluate_all_layers


LAYER_SCAN_SCHEMA = "spec_gap.layer_scan.v1"
EVALUATION_METHOD = "leave_one_match_group_out"
NEGATIVE_CONTROL_AUROC_GUARDRAIL = 0.75


def run_construction_layer_scan(
    index_rows: Iterable[dict[str, Any]],
    *,
    label_target: str = "injection_present",
    control_audit: dict[str, Any] | None = None,
    verify_checksums: bool = True,
    random_state: int = 42,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Evaluate separate all-layer probes for each mode, agent, and checkpoint.

    Related 2-hop and 3-hop trajectories stay together because cross-validation
    holds out the full match group. With only two groups, the result is a
    pipeline and signal check rather than a stable estimate of generalization.
    """

    rows = list(index_rows)
    if not rows:
        raise ValueError("At least one activation-index row is required.")
    if label_target not in LABEL_TARGETS:
        raise ValueError(
            f"label_target must be one of {sorted(LABEL_TARGETS)}, got {label_target!r}."
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("thinking_mode")),
            str(row.get("agent_id")),
            str(row.get("checkpoint")),
        )
        grouped[key].append(row)

    strata = []
    for (thinking_mode, agent_id, checkpoint), cohort in sorted(grouped.items()):
        cohort = sorted(
            cohort,
            key=lambda row: (
                row["match_group_id"],
                row["delegation_depth"],
                row["treatment"],
                row["trajectory_id"],
            ),
        )
        base = _stratum_metadata(
            cohort,
            thinking_mode=thinking_mode,
            agent_id=agent_id,
            checkpoint=checkpoint,
            label_target=label_target,
        )
        skip_reason = _skip_reason(cohort, label_target=label_target)
        if skip_reason:
            strata.append({**base, "status": "skipped", "skip_reason": skip_reason})
            continue

        batch = load_probe_activation_batch(
            cohort,
            label_target=label_target,
            verify_checksums=verify_checksums,
        )
        groups = np.asarray([
            row["match_group_id"] for row in batch.metadata
        ])
        results = evaluate_all_layers(
            batch.activations,
            batch.labels,
            metadata={"scenario_idx": groups},
            leave_scenario_out=True,
            max_iter=max_iter,
            random_state=random_state,
        )
        layer_results = {}
        for layer, result in sorted(results.layer_results.items()):
            layer_results[str(layer)] = {
                "auroc_mean": result.auroc_mean,
                "auroc_std": result.auroc_std,
                "auroc_per_fold": [float(value) for value in result.auroc_per_fold],
                "accuracy_mean": result.accuracy_mean,
                "accuracy_std": result.accuracy_std,
                "brier_mean": result.brier_mean,
                "brier_std": result.brier_std,
                "ece_mean": result.ece_mean,
                "ece_std": result.ece_std,
                "diff_in_means_norm": float(np.linalg.norm(
                    results.diff_in_means[layer]
                )),
            }
        best_layer = min(
            layer_results,
            key=lambda layer: (
                -_finite_or_negative_infinity(layer_results[layer]["auroc_mean"]),
                int(layer),
            ),
        )
        strata.append({
            **base,
            "status": "completed",
            "best_layer_by_mean_auroc": int(best_layer),
            "best_mean_auroc": layer_results[best_layer]["auroc_mean"],
            "layer_results": layer_results,
        })

    completed = [stratum for stratum in strata if stratum["status"] == "completed"]
    negative_control = (
        _planner_negative_control_from_audit(strata, control_audit)
        if control_audit is not None
        else _planner_negative_control(rows, strata)
    )
    _annotate_strata_with_controls(strata, negative_control)
    layer_selection_blockers = ["only_two_independent_match_groups"]
    if negative_control["blocks_layer_selection"]:
        layer_selection_blockers.append("failed_strict_planner_input_control")
    if negative_control.get("stochastic_output_null_not_calibrated"):
        layer_selection_blockers.append("stochastic_output_null_not_calibrated")
    return {
        "schema_version": LAYER_SCAN_SCHEMA,
        "probe_name": "goldowsky_dill_logistic",
        "label_target": label_target,
        "evaluation_method": EVALUATION_METHOD,
        "selection_status": "exploratory_small_n",
        "layer_selection_allowed": False,
        "layer_selection_blockers": layer_selection_blockers,
        "claim_scope": (
            "Construction-label signal check only. This is not behavioral-compromise "
            "detection and is not a stable layer-selection estimate."
        ),
        "index_rows_considered": len(rows),
        "match_groups": sorted({str(row["match_group_id"]) for row in rows}),
        "completed_strata": len(completed),
        "skipped_strata": len(strata) - len(completed),
        "pre_injection_negative_control": negative_control,
        "strata": strata,
        "limitations": [
            "Only two independent match groups are available.",
            "The same task appears at both depths inside each held-out match group.",
            "Layer rankings are descriptive and must not be reported as final AUROC.",
            "All injected trajectories resisted, so action-executed labels have no positive class.",
            "Planner generated-token checkpoints require a calibrated stochastic null.",
            "A failed strict planner input control blocks data-driven layer selection.",
        ],
    }


def layer_scan_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten completed scan results for CSV output and plotting."""

    rows = []
    for stratum in result.get("strata", []):
        if stratum.get("status") != "completed":
            continue
        for layer, metrics in stratum["layer_results"].items():
            rows.append({
                "probe_name": result["probe_name"],
                "label_target": result["label_target"],
                "evaluation_method": result["evaluation_method"],
                "thinking_mode": stratum["thinking_mode"],
                "agent_id": stratum["agent_id"],
                "agent_role": stratum["agent_role"],
                "checkpoint": stratum["checkpoint"],
                "layer": int(layer),
                "sample_count": stratum["sample_count"],
                "match_group_count": stratum["match_group_count"],
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "auroc_per_fold"
                },
            })
    return rows


def _stratum_metadata(
    cohort: list[dict[str, Any]],
    *,
    thinking_mode: str,
    agent_id: str,
    checkpoint: str,
    label_target: str,
) -> dict[str, Any]:
    labels = [row.get("labels", {}).get(label_target) for row in cohort]
    return {
        "thinking_mode": thinking_mode,
        "agent_id": agent_id,
        "agent_role": str(cohort[0].get("agent_role")),
        "checkpoint": checkpoint,
        "sample_count": len(cohort),
        "match_group_count": len({row["match_group_id"] for row in cohort}),
        "match_groups": sorted({str(row["match_group_id"]) for row in cohort}),
        "delegation_depths": sorted({str(row["delegation_depth"]) for row in cohort}),
        "class_counts": {
            "negative": sum(label == 0 for label in labels),
            "positive": sum(label == 1 for label in labels),
            "missing": sum(label not in (0, 1) for label in labels),
        },
    }


def _skip_reason(
    cohort: list[dict[str, Any]],
    *,
    label_target: str,
) -> str | None:
    labels = [row.get("labels", {}).get(label_target) for row in cohort]
    if any(label not in (0, 1) for label in labels):
        return f"Some samples do not have a binary {label_target!r} label."
    if len(set(labels)) < 2:
        return f"The {label_target!r} label has only one class."
    groups = sorted({str(row["match_group_id"]) for row in cohort})
    if len(groups) < 2:
        return "Leave-one-match-group-out evaluation requires at least two groups."
    for held_out in groups:
        train_labels = {
            row["labels"][label_target]
            for row in cohort
            if str(row["match_group_id"]) != held_out
        }
        test_labels = {
            row["labels"][label_target]
            for row in cohort
            if str(row["match_group_id"]) == held_out
        }
        if train_labels != {0, 1} or test_labels != {0, 1}:
            return (
                "Every held-out fold must contain both classes in its training and "
                "test partitions."
            )
    return None


def _finite_or_negative_infinity(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def _planner_negative_control(
    rows: list[dict[str, Any]],
    strata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check for false separation before Worker 1 receives any document."""

    planner_rows = [row for row in rows if row.get("agent_id") == "planner_1"]
    planner_strata = [
        stratum
        for stratum in strata
        if stratum.get("agent_id") == "planner_1"
        and stratum.get("status") == "completed"
    ]
    if not planner_rows or not planner_strata:
        return {
            "status": "not_available",
            "blocks_layer_selection": True,
            "reason": "No completed planner stratum is available as a negative control.",
        }

    unique_pair_hashes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in planner_rows:
        key = (str(row["thinking_mode"]), str(row["matched_pair_id"]))
        unique_pair_hashes[key][str(row["treatment"])] = str(
            row["rendered_prompt_hash"]
        )

    checkpoint_controls = []
    for stratum in sorted(
        planner_strata,
        key=lambda item: (item["thinking_mode"], item["checkpoint"]),
    ):
        mode = str(stratum["thinking_mode"])
        checkpoint = str(stratum["checkpoint"])
        control_rows = [
            row
            for row in planner_rows
            if str(row["thinking_mode"]) == mode
            and str(row["checkpoint"]) == checkpoint
        ]
        pair_hashes: dict[str, dict[str, str]] = defaultdict(dict)
        for row in control_rows:
            pair_hashes[str(row["matched_pair_id"])][str(row["treatment"])] = str(
                row["rendered_prompt_hash"]
            )
        incomplete_pairs = []
        prompt_mismatches = []
        for pair_id, by_treatment in sorted(pair_hashes.items()):
            if set(by_treatment) != {"clean", "injected"}:
                incomplete_pairs.append(pair_id)
            elif by_treatment["clean"] != by_treatment["injected"]:
                prompt_mismatches.append(pair_id)
        strongest_auroc = float(stratum["best_mean_auroc"])
        false_separation = strongest_auroc >= NEGATIVE_CONTROL_AUROC_GUARDRAIL
        design_invalid = bool(incomplete_pairs or prompt_mismatches)
        if design_invalid:
            status = "invalid_control_pairs"
            reason = (
                "Planner clean/injected pairs must have complete, identical "
                "rendered prompts."
            )
        elif false_separation:
            status = "failed_spurious_separation"
            reason = (
                "This pre-injection checkpoint separates construction labels above "
                "the descriptive guardrail."
            )
        else:
            status = "passed"
            reason = (
                "This planner checkpoint stayed below the descriptive guardrail."
            )
        checkpoint_controls.append({
            "thinking_mode": mode,
            "checkpoint": checkpoint,
            "status": status,
            "blocks_exploratory_ranking": design_invalid or false_separation,
            "reason": reason,
            "matched_prompt_pair_count": len(pair_hashes),
            "incomplete_pairs": incomplete_pairs,
            "prompt_hash_mismatches": prompt_mismatches,
            "strongest_layer": int(stratum["best_layer_by_mean_auroc"]),
            "strongest_mean_auroc": strongest_auroc,
        })

    strongest = max(
        checkpoint_controls,
        key=lambda control: _finite_or_negative_infinity(
            control["strongest_mean_auroc"]
        ),
    )
    all_incomplete = sorted({
        pair_id
        for control in checkpoint_controls
        for pair_id in control["incomplete_pairs"]
    })
    all_mismatches = sorted({
        pair_id
        for control in checkpoint_controls
        for pair_id in control["prompt_hash_mismatches"]
    })
    any_invalid = any(
        control["status"] == "invalid_control_pairs"
        for control in checkpoint_controls
    )
    any_failed = any(
        control["status"] == "failed_spurious_separation"
        for control in checkpoint_controls
    )
    if any_invalid:
        status = "invalid_control_pairs"
        reason = (
            "Planner clean/injected pairs must have complete, identical rendered prompts."
        )
    elif any_failed:
        status = "failed_spurious_separation"
        reason = (
            "At least one pre-injection mode/checkpoint separates construction "
            "labels above the descriptive guardrail."
        )
    else:
        status = "passed"
        reason = "Planner construction-label AUROC stayed below the guardrail."
    return {
        "status": status,
        "blocks_layer_selection": any_invalid or any_failed,
        "reason": reason,
        "expected_signal": "none_before_document_retrieval",
        "descriptive_auroc_guardrail": NEGATIVE_CONTROL_AUROC_GUARDRAIL,
        "matched_prompt_pair_count": len(unique_pair_hashes),
        "incomplete_pairs": all_incomplete,
        "prompt_hash_mismatches": all_mismatches,
        "checkpoint_controls": checkpoint_controls,
        "qualified_mode_checkpoints": [
            {
                "thinking_mode": control["thinking_mode"],
                "checkpoint": control["checkpoint"],
            }
            for control in checkpoint_controls
            if control["status"] == "passed"
        ],
        "blocked_mode_checkpoints": [
            {
                "thinking_mode": control["thinking_mode"],
                "checkpoint": control["checkpoint"],
            }
            for control in checkpoint_controls
            if control["status"] != "passed"
        ],
        "strongest_observed_stratum": {
            "thinking_mode": strongest["thinking_mode"],
            "checkpoint": strongest["checkpoint"],
            "layer": strongest["strongest_layer"],
            "mean_auroc": strongest["strongest_mean_auroc"],
        },
    }


def _planner_negative_control_from_audit(
    strata: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Translate paired identity results into layer-scan qualification rules."""

    strict = audit.get("strict_planner_input_control", {})
    stochastic = audit.get("stochastic_planner_output_control", {})
    mode_controls = {
        str(control.get("thinking_mode")): control
        for control in strict.get("mode_controls", [])
    }
    planner_strata = [
        stratum
        for stratum in strata
        if stratum.get("agent_id") == "planner_1"
        and stratum.get("status") == "completed"
    ]
    if not planner_strata:
        return {
            "status": "not_available",
            "blocks_layer_selection": True,
            "reason": "No completed planner stratum is available.",
            "checkpoint_controls": [],
            "qualified_mode_checkpoints": [],
            "blocked_mode_checkpoints": [],
        }

    checkpoint_controls = []
    for stratum in sorted(
        planner_strata,
        key=lambda item: (item["thinking_mode"], item["checkpoint"]),
    ):
        mode = str(stratum["thinking_mode"])
        checkpoint = str(stratum["checkpoint"])
        if checkpoint == "last_input_token":
            mode_control = mode_controls.get(mode)
            passed = bool(mode_control and mode_control.get("status") == "passed")
            status = (
                "passed_strict_input_control"
                if passed
                else "failed_strict_input_control"
            )
            reason = (
                "Identical planner prompts and input tokens produced matching "
                "last-input activations."
                if passed
                else "The paired planner last-input identity audit failed."
            )
            blocks = not passed
        else:
            status = "stochastic_null_uncalibrated"
            reason = (
                "This checkpoint follows sampled planner generation. Its AUROC "
                "must be compared with an empirical stochastic null."
            )
            blocks = True
        checkpoint_controls.append({
            "thinking_mode": mode,
            "checkpoint": checkpoint,
            "status": status,
            "blocks_exploratory_ranking": blocks,
            "reason": reason,
            "strongest_layer": int(stratum["best_layer_by_mean_auroc"]),
            "strongest_mean_auroc": float(stratum["best_mean_auroc"]),
        })

    strict_failed = strict.get("status") != "passed"
    qualified_statuses = {"passed", "passed_strict_input_control"}
    return {
        "status": (
            "failed_strict_input_control" if strict_failed else "strict_input_passed"
        ),
        "blocks_layer_selection": strict_failed,
        "reason": str(strict.get("reason", "Strict planner audit unavailable.")),
        "expected_signal": "none_at_planner_last_input",
        "strict_input_control": strict,
        "stochastic_output_control": stochastic,
        "stochastic_output_null_not_calibrated": bool(
            stochastic.get("blocks_stratum_ranking_until_calibrated")
        ),
        "checkpoint_controls": checkpoint_controls,
        "qualified_mode_checkpoints": [
            {
                "thinking_mode": control["thinking_mode"],
                "checkpoint": control["checkpoint"],
            }
            for control in checkpoint_controls
            if control["status"] in qualified_statuses
        ],
        "blocked_mode_checkpoints": [
            {
                "thinking_mode": control["thinking_mode"],
                "checkpoint": control["checkpoint"],
            }
            for control in checkpoint_controls
            if control["status"] not in qualified_statuses
        ],
    }


def _annotate_strata_with_controls(
    strata: list[dict[str, Any]],
    negative_control: dict[str, Any],
) -> None:
    controls = {
        (control["thinking_mode"], control["checkpoint"]): control
        for control in negative_control.get("checkpoint_controls", [])
    }
    for stratum in strata:
        control = controls.get((stratum["thinking_mode"], stratum["checkpoint"]))
        if control is None:
            stratum["negative_control_status"] = "not_available"
            stratum["exploratory_ranking_qualified"] = False
        else:
            stratum["negative_control_status"] = control["status"]
            stratum["exploratory_ranking_qualified"] = control["status"] in {
                "passed",
                "passed_strict_input_control",
            }
