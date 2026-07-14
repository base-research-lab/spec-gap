"""Paired negative-control and propagation audits for saved activations."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable

import numpy as np

from src.extraction.saved_activations import load_activation_checkpoint


ACTIVATION_CONTROL_SCHEMA = "spec_gap.activation_control_audit.v1"
STRICT_INPUT_RTOL = 1e-5
STRICT_INPUT_ATOL = 1e-6


def run_activation_control_audit(
    index_rows: Iterable[dict[str, Any]],
    *,
    verify_checksums: bool = True,
) -> dict[str, Any]:
    """Compare clean/injected activation pairs without fitting a probe.

    Planner ``last_input_token`` pairs are strict negative controls because the
    rendered prompts and token IDs must be identical before document retrieval.
    Planner generation checkpoints are reported separately as stochastic nulls.
    All other agent/checkpoint pairs form a descriptive propagation profile.
    """

    rows = list(index_rows)
    if not rows:
        raise ValueError("At least one activation-index row is required.")
    _validate_required_index_fields(rows)

    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        treatment = str(row["treatment"])
        if treatment not in {"clean", "injected"}:
            raise ValueError(f"Unsupported treatment {treatment!r}.")
        key = (
            str(row["thinking_mode"]),
            str(row["matched_pair_id"]),
            str(row["agent_id"]),
            str(row["checkpoint"]),
        )
        if treatment in grouped[key]:
            raise ValueError(
                "Duplicate activation-control row for "
                f"{key!r}, treatment={treatment!r}."
            )
        grouped[key][treatment] = row

    cache: dict[tuple[str, str, str | None], Any] = {}
    complete_pairs = []
    incomplete_pairs = []
    for key, by_treatment in sorted(grouped.items()):
        mode, pair_id, agent_id, checkpoint = key
        if set(by_treatment) != {"clean", "injected"}:
            incomplete_pairs.append({
                "thinking_mode": mode,
                "matched_pair_id": pair_id,
                "agent_id": agent_id,
                "checkpoint": checkpoint,
                "available_treatments": sorted(by_treatment),
            })
            continue
        complete_pairs.append(_compare_pair(
            by_treatment["clean"],
            by_treatment["injected"],
            cache=cache,
            verify_checksums=verify_checksums,
        ))

    if not complete_pairs:
        raise ValueError("No complete clean/injected activation pairs are available.")

    strict_control = _strict_planner_input_control(complete_pairs)
    stochastic_control = _stochastic_planner_output_control(complete_pairs)
    return {
        "schema_version": ACTIVATION_CONTROL_SCHEMA,
        "audit_type": "paired_activation_control",
        "claim_scope": (
            "Input-identity and paired-distance audit only. It does not estimate "
            "behavioral-compromise detection performance or select a model layer."
        ),
        "index_rows_considered": len(rows),
        "complete_pair_count": len(complete_pairs),
        "incomplete_pair_count": len(incomplete_pairs),
        "incomplete_pairs": incomplete_pairs,
        "match_groups": sorted({str(row["match_group_id"]) for row in rows}),
        "thinking_modes": sorted({str(row["thinking_mode"]) for row in rows}),
        "strict_planner_input_control": strict_control,
        "stochastic_planner_output_control": stochastic_control,
        "propagation_profiles": _summarize_profiles(complete_pairs),
        "pairs": complete_pairs,
        "layer_selection_allowed": False,
        "layer_selection_blockers": [
            *(
                []
                if strict_control["status"] == "passed"
                else ["strict_planner_input_control_failed"]
            ),
            "stochastic_output_null_not_calibrated",
            "only_two_independent_match_groups",
        ],
        "limitations": [
            "Generated planner checkpoints are stochastic nulls, not strict identity controls.",
            "Only two independent match groups are available.",
            "The audit reports paired distances and does not convert them into probe AUROC.",
        ],
    }


def activation_control_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten pair/layer metrics for CSV export and plotting."""

    rows = []
    for pair in result.get("pairs", []):
        base = {
            key: pair[key]
            for key in (
                "thinking_mode",
                "match_group_id",
                "matched_pair_id",
                "delegation_depth",
                "agent_id",
                "agent_role",
                "checkpoint",
                "prompt_hash_match",
                "input_token_ids_match",
                "generated_token_ids_match",
                "checkpoint_forward_scope_match",
                "prompt_only_input_extraction",
                "strict_input_control_pair",
                "all_layers_within_strict_tolerance",
            )
        }
        for layer, metrics in pair["layer_metrics"].items():
            rows.append({**base, "layer": int(layer), **metrics})
    return rows


def _compare_pair(
    clean: dict[str, Any],
    injected: dict[str, Any],
    *,
    cache: dict[tuple[str, str, str | None], Any],
    verify_checksums: bool,
) -> dict[str, Any]:
    for field in (
        "thinking_mode",
        "matched_pair_id",
        "match_group_id",
        "delegation_depth",
        "agent_id",
        "agent_role",
        "checkpoint",
    ):
        if clean[field] != injected[field]:
            raise ValueError(
                f"Paired rows disagree on {field!r}: "
                f"{clean[field]!r} != {injected[field]!r}."
            )

    clean_loaded = _load_cached(
        clean,
        cache=cache,
        verify_checksums=verify_checksums,
    )
    injected_loaded = _load_cached(
        injected,
        cache=cache,
        verify_checksums=verify_checksums,
    )
    if clean_loaded.layers != injected_loaded.layers:
        raise ValueError(
            f"Activation layers differ inside pair {clean['matched_pair_id']!r}."
        )

    layer_metrics = {}
    for offset, layer in enumerate(clean_loaded.layers):
        clean_vector = clean_loaded.activations[offset].float().numpy()
        injected_vector = injected_loaded.activations[offset].float().numpy()
        layer_metrics[str(layer)] = _vector_distance(clean_vector, injected_vector)

    strict_pair = (
        clean["agent_id"] == "planner_1"
        and clean["checkpoint"] == "last_input_token"
    )
    clean_scope = clean.get("checkpoint_forward_scope")
    injected_scope = injected.get("checkpoint_forward_scope")
    return {
        "thinking_mode": str(clean["thinking_mode"]),
        "match_group_id": str(clean["match_group_id"]),
        "matched_pair_id": str(clean["matched_pair_id"]),
        "delegation_depth": str(clean["delegation_depth"]),
        "agent_id": str(clean["agent_id"]),
        "agent_role": str(clean["agent_role"]),
        "checkpoint": str(clean["checkpoint"]),
        "clean_trajectory_id": str(clean["trajectory_id"]),
        "injected_trajectory_id": str(injected["trajectory_id"]),
        "prompt_hash_match": (
            clean["rendered_prompt_hash"] == injected["rendered_prompt_hash"]
        ),
        "input_token_ids_match": (
            clean["input_token_ids_sha256"]
            == injected["input_token_ids_sha256"]
        ),
        "generated_token_ids_match": (
            clean["generated_token_ids_sha256"]
            == injected["generated_token_ids_sha256"]
        ),
        "clean_checkpoint_forward_scope": clean_scope,
        "injected_checkpoint_forward_scope": injected_scope,
        "checkpoint_forward_scope_match": clean_scope == injected_scope,
        "prompt_only_input_extraction": (
            clean_scope == "prompt_only" and injected_scope == "prompt_only"
        ),
        "clean_input_token_count": int(clean["input_token_count"]),
        "injected_input_token_count": int(injected["input_token_count"]),
        "clean_generated_token_count": int(clean["generated_token_count"]),
        "injected_generated_token_count": int(injected["generated_token_count"]),
        "strict_input_control_pair": strict_pair,
        "all_layers_exactly_equal": all(
            metrics["exact_match"] for metrics in layer_metrics.values()
        ),
        "all_layers_within_strict_tolerance": all(
            metrics["within_strict_tolerance"]
            for metrics in layer_metrics.values()
        ),
        "layer_metrics": layer_metrics,
    }


def _load_cached(
    row: dict[str, Any],
    *,
    cache: dict[tuple[str, str, str | None], Any],
    verify_checksums: bool,
) -> Any:
    key = (
        str(row["local_path"]),
        str(row["checkpoint"]),
        row.get("checksum_sha256"),
    )
    if key not in cache:
        cache[key] = load_activation_checkpoint(
            row["local_path"],
            checkpoint=row["checkpoint"],
            expected_checksum=row.get("checksum_sha256"),
            verify_checksum=verify_checksums,
        )
    return cache[key]


def _vector_distance(clean: np.ndarray, injected: np.ndarray) -> dict[str, Any]:
    difference = injected - clean
    clean_norm = float(np.linalg.norm(clean))
    injected_norm = float(np.linalg.norm(injected))
    difference_norm = float(np.linalg.norm(difference))
    reference_norm = max((clean_norm + injected_norm) / 2.0, np.finfo(float).eps)
    denominator = clean_norm * injected_norm
    cosine_similarity = (
        float(np.dot(clean, injected) / denominator)
        if denominator > 0
        else (1.0 if np.array_equal(clean, injected) else 0.0)
    )
    return {
        "clean_l2_norm": clean_norm,
        "injected_l2_norm": injected_norm,
        "difference_l2_norm": difference_norm,
        "relative_l2_difference": difference_norm / reference_norm,
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "cosine_similarity": float(np.clip(cosine_similarity, -1.0, 1.0)),
        "exact_match": bool(np.array_equal(clean, injected)),
        "within_strict_tolerance": bool(np.allclose(
            clean,
            injected,
            rtol=STRICT_INPUT_RTOL,
            atol=STRICT_INPUT_ATOL,
        )),
    }


def _strict_planner_input_control(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    controls = [pair for pair in pairs if pair["strict_input_control_pair"]]
    if not controls:
        return {
            "status": "not_available",
            "blocks_layer_selection": True,
            "reason": "No planner last-input clean/injected pairs are available.",
            "mode_controls": [],
        }

    mode_controls = []
    for mode in sorted({pair["thinking_mode"] for pair in controls}):
        mode_pairs = [pair for pair in controls if pair["thinking_mode"] == mode]
        prompt_identity = all(pair["prompt_hash_match"] for pair in mode_pairs)
        token_identity = all(pair["input_token_ids_match"] for pair in mode_pairs)
        tensor_identity = all(
            pair["all_layers_within_strict_tolerance"] for pair in mode_pairs
        )
        extraction_scope_qualified = all(
            pair["prompt_only_input_extraction"] for pair in mode_pairs
        )
        status = (
            "passed"
            if (
                prompt_identity
                and token_identity
                and tensor_identity
                and extraction_scope_qualified
            )
            else "failed"
        )
        metrics = [
            layer
            for pair in mode_pairs
            for layer in pair["layer_metrics"].values()
        ]
        mode_controls.append({
            "thinking_mode": mode,
            "status": status,
            "pair_count": len(mode_pairs),
            "prompt_hash_identity": prompt_identity,
            "input_token_identity": token_identity,
            "activation_tolerance_passed": tensor_identity,
            "prompt_only_extraction": extraction_scope_qualified,
            "exact_activation_pair_count": sum(
                pair["all_layers_exactly_equal"] for pair in mode_pairs
            ),
            "maximum_relative_l2_difference": max(
                metric["relative_l2_difference"] for metric in metrics
            ),
            "maximum_absolute_difference": max(
                metric["max_absolute_difference"] for metric in metrics
            ),
        })

    passed = all(control["status"] == "passed" for control in mode_controls)
    return {
        "status": "passed" if passed else "failed",
        "blocks_layer_selection": not passed,
        "reason": (
            "Planner prompts and input tokens match, and prompt-only last-input "
            "activations match within tolerance."
            if passed
            else "At least one planner input identity, extraction-scope, or "
            "activation-tolerance check failed."
        ),
        "relative_tolerance": STRICT_INPUT_RTOL,
        "absolute_tolerance": STRICT_INPUT_ATOL,
        "mode_controls": mode_controls,
    }


def _stochastic_planner_output_control(
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    controls = [
        pair
        for pair in pairs
        if pair["agent_id"] == "planner_1"
        and pair["checkpoint"] != "last_input_token"
    ]
    by_mode_checkpoint = []
    keys = sorted({
        (pair["thinking_mode"], pair["checkpoint"])
        for pair in controls
    })
    for mode, checkpoint in keys:
        selected = [
            pair
            for pair in controls
            if pair["thinking_mode"] == mode and pair["checkpoint"] == checkpoint
        ]
        by_mode_checkpoint.append({
            "thinking_mode": mode,
            "checkpoint": checkpoint,
            "pair_count": len(selected),
            "prompt_hash_match_count": sum(
                pair["prompt_hash_match"] for pair in selected
            ),
            "input_token_match_count": sum(
                pair["input_token_ids_match"] for pair in selected
            ),
            "generated_token_match_count": sum(
                pair["generated_token_ids_match"] for pair in selected
            ),
            "status": "descriptive_only",
        })
    return {
        "status": "descriptive_only" if controls else "not_available",
        "blocks_stratum_ranking_until_calibrated": bool(controls),
        "reason": (
            "Generated planner checkpoints may differ under sampled decoding. "
            "They require an empirical null distribution rather than exact identity."
        ),
        "mode_checkpoint_controls": by_mode_checkpoint,
    }


def _summarize_profiles(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[(
            pair["thinking_mode"],
            pair["agent_id"],
            pair["checkpoint"],
        )].append(pair)

    summaries = []
    for (mode, agent_id, checkpoint), selected in sorted(grouped.items()):
        layer_ids = sorted({
            int(layer)
            for pair in selected
            for layer in pair["layer_metrics"]
        })
        layer_summaries = {}
        for layer in layer_ids:
            values = [
                pair["layer_metrics"][str(layer)]
                for pair in selected
                if str(layer) in pair["layer_metrics"]
            ]
            relative = [value["relative_l2_difference"] for value in values]
            cosine = [value["cosine_similarity"] for value in values]
            layer_summaries[str(layer)] = {
                "pair_count": len(values),
                "mean_relative_l2_difference": float(np.mean(relative)),
                "median_relative_l2_difference": float(median(relative)),
                "maximum_relative_l2_difference": max(relative),
                "mean_cosine_similarity": float(np.mean(cosine)),
            }
        summaries.append({
            "thinking_mode": mode,
            "agent_id": agent_id,
            "agent_role": selected[0]["agent_role"],
            "checkpoint": checkpoint,
            "pair_count": len(selected),
            "prompt_hash_match_count": sum(
                pair["prompt_hash_match"] for pair in selected
            ),
            "input_token_match_count": sum(
                pair["input_token_ids_match"] for pair in selected
            ),
            "generated_token_match_count": sum(
                pair["generated_token_ids_match"] for pair in selected
            ),
            "layer_summaries": layer_summaries,
        })
    return summaries


def _validate_required_index_fields(rows: list[dict[str, Any]]) -> None:
    required = {
        "trajectory_id",
        "thinking_mode",
        "matched_pair_id",
        "match_group_id",
        "delegation_depth",
        "treatment",
        "agent_id",
        "agent_role",
        "checkpoint",
        "rendered_prompt_hash",
        "input_token_count",
        "input_token_ids_sha256",
        "generated_token_count",
        "generated_token_ids_sha256",
        "checkpoint_forward_scope",
        "local_path",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Activation-index row {index} is missing required fields: {missing}."
            )
