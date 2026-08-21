"""Group-safe per-step probe scoring for Scenario 1 activations.

The rows produced here are scores from representation probes, not new language
model generations and not inferred behavioral outcomes.  Every score is made
for a held-out match group using a model fitted only on the other groups.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.extraction.saved_activations import (
    load_probe_activation_batch,
    normalize_index_analysis_tier,
)
from src.probes.lat_baseline import ContrastPairLATProbe
from src.probes.linear_probe import make_probe_pipeline


PER_STEP_SCORE_SCHEMA = "spec_gap.per_step_probe_score.v2"
EVALUATION_METHOD = "leave_one_match_group_out"
SUPPORTED_CHECKPOINT = "last_input_token"
PROBE_NAMES = ("goldowsky_dill_logistic", "lat_contrast_pair_pca")
PROBE_SCORE_LABEL_TARGETS = {"injection_present"}
ACTIVATION_PREPROCESSING_METHODS = {
    "none",
    "train_domain_mean_residualized",
}
BEHAVIORAL_OUTCOMES = {
    "clean",
    "resisted",
    "propagated_but_not_executed",
    "attempted_but_blocked",
    "executed",
    "indeterminate",
}


@dataclass(frozen=True)
class MatchGroupDesign:
    """Design metadata absent from the activation index."""

    wording_id: str
    seed: int


def load_match_group_designs(path: str | Path) -> dict[str, MatchGroupDesign]:
    """Load one wording and seed declaration for each manifest match group."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load Scenario 1 manifest {source}: {error}"
        ) from error
    trajectories = payload.get("trajectories") if isinstance(payload, dict) else None
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError(
            "Scenario 1 manifest must contain a non-empty trajectories list."
        )

    designs: dict[str, MatchGroupDesign] = {}
    for index, entry in enumerate(trajectories):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest trajectory {index} must be an object.")
        group_id = entry.get("group_id")
        wording = entry.get("wording")
        seed = entry.get("seed")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"Manifest trajectory {index} has no group_id.")
        if not isinstance(wording, str) or not wording:
            raise ValueError(f"Manifest trajectory {index} has no wording.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"Manifest trajectory {index} has an invalid seed.")
        design = MatchGroupDesign(wording_id=wording, seed=seed)
        previous = designs.setdefault(group_id, design)
        if previous != design:
            raise ValueError(
                f"Manifest group {group_id!r} mixes wording or seed declarations."
            )
    return designs


def generate_per_step_probe_scores(
    index_rows: Iterable[dict[str, Any]],
    *,
    match_group_designs: dict[str, MatchGroupDesign],
    label_target: str = "injection_present",
    checkpoint: str = SUPPORTED_CHECKPOINT,
    layers: Sequence[int] | None = None,
    verify_checksums: bool = True,
    random_state: int = 42,
    max_iter: int = 1000,
    activation_preprocessing: str = "none",
) -> list[dict[str, Any]]:
    """Create held-out Goldowsky-Dill and LAT scores for every agent step.

    Probes are fitted separately for each thinking mode, agent identity, and
    layer.  Related 2-hop/3-hop and clean/injected trajectories are kept in the
    same held-out match-group fold.
    """

    if label_target not in PROBE_SCORE_LABEL_TARGETS:
        raise ValueError(
            "label_target must be one of "
            f"{sorted(PROBE_SCORE_LABEL_TARGETS)}, got {label_target!r}. "
            "Behavioral-action probes require mixed executed and non-executed outcomes "
            "and are not supported by the current construction-diagnostic sample."
        )
    if checkpoint != SUPPORTED_CHECKPOINT:
        raise ValueError(
            "Only last_input_token is qualified for the current controlled analysis."
        )
    if activation_preprocessing not in ACTIVATION_PREPROCESSING_METHODS:
        raise ValueError(
            "activation_preprocessing must be one of "
            f"{sorted(ACTIVATION_PREPROCESSING_METHODS)}, got "
            f"{activation_preprocessing!r}."
        )

    rows = [dict(row) for row in index_rows if row.get("checkpoint") == checkpoint]
    if not rows:
        raise ValueError(f"No activation-index rows use checkpoint {checkpoint!r}.")
    _validate_index_design(rows, match_group_designs, label_target=label_target)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["thinking_mode"]), str(row["agent_id"]))].append(row)

    score_rows: list[dict[str, Any]] = []
    for (thinking_mode, agent_id), cohort in sorted(grouped.items()):
        cohort = sorted(cohort, key=_index_row_sort_key)
        batch = load_probe_activation_batch(
            cohort,
            label_target=label_target,
            layers=layers,
            verify_checksums=verify_checksums,
        )
        groups = np.asarray(
            [str(metadata["match_group_id"]) for metadata in batch.metadata]
        )
        pair_ids = np.asarray(
            [str(metadata["matched_pair_id"]) for metadata in batch.metadata]
        )
        unique_groups = sorted(set(groups.tolist()))
        if len(unique_groups) < 2:
            raise ValueError(
                f"{thinking_mode}/{agent_id} requires at least two match groups."
            )

        for layer, activations in sorted(batch.activations.items()):
            for held_out_group in unique_groups:
                test_mask = groups == held_out_group
                train_mask = ~test_mask
                y_train = batch.labels[train_mask]
                y_test = batch.labels[test_mask]
                if set(y_train.tolist()) != {0, 1}:
                    raise ValueError(
                        f"Training fold {held_out_group!r} lacks both label classes for "
                        f"{thinking_mode}/{agent_id}/layer-{layer}."
                    )
                if set(y_test.tolist()) != {0, 1}:
                    raise ValueError(
                        f"Held-out fold {held_out_group!r} lacks both label classes for "
                        f"{thinking_mode}/{agent_id}/layer-{layer}."
                    )

                X_train, X_test = preprocess_activation_fold(
                    activations[train_mask],
                    activations[test_mask],
                    train_groups=groups[train_mask],
                    method=activation_preprocessing,
                )

                probe_outputs = _fit_fold_probes(
                    X_train,
                    y_train,
                    pair_ids[train_mask],
                    X_test,
                    random_state=random_state,
                    max_iter=max_iter,
                )
                test_indices = np.flatnonzero(test_mask)
                for probe_name, (probabilities, fit_status) in probe_outputs.items():
                    for batch_index, probability in zip(test_indices, probabilities):
                        metadata = batch.metadata[int(batch_index)]
                        score_rows.append(
                            _score_row(
                                metadata,
                                score=float(probability),
                                probe_name=probe_name,
                                layer=int(layer),
                                label_target=label_target,
                                held_out_group=held_out_group,
                                fit_status=fit_status,
                                design=match_group_designs[
                                    str(metadata["match_group_id"])
                                ],
                                activation_preprocessing=activation_preprocessing,
                            )
                        )

    return sorted(score_rows, key=_score_row_sort_key)


def summarize_per_step_probe_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return coverage and method metadata for a score artifact."""

    if not rows:
        raise ValueError("At least one per-step probe score is required.")
    analysis_tiers = {
        normalize_index_analysis_tier(row.get("analysis_tier") or "unclassified")
        for row in rows
    }
    if len(analysis_tiers) != 1:
        raise ValueError("Per-step probe scores must contain one analysis tier.")
    return {
        "schema_version": PER_STEP_SCORE_SCHEMA,
        "artifact_kind": "held_out_per_step_probe_scores",
        "analysis_tier": next(iter(analysis_tiers)),
        "claim_scope": (
            "Construction-label diagnostic only. The scores are not generated model "
            "responses or behavioral ground truth."
        ),
        "evaluation_method": EVALUATION_METHOD,
        "checkpoint": sorted({str(row["checkpoint"]) for row in rows}),
        "label_targets": sorted({str(row["label_target"]) for row in rows}),
        "score_rows": len(rows),
        "trajectories": len({str(row["trajectory_id"]) for row in rows}),
        "agent_steps": len(
            {(str(row["trajectory_id"]), int(row["hop_index"])) for row in rows}
        ),
        "match_groups": sorted({str(row["match_group_id"]) for row in rows}),
        "thinking_modes": sorted({str(row["thinking_mode"]) for row in rows}),
        "probes": sorted({str(row["probe_name"]) for row in rows}),
        "layers": sorted({int(row["layer"]) for row in rows}),
        "prespecified_reference_layer": 40,
        "prespecified_ablation_layers": [32, 48],
        "layer_selection_note": (
            "Layer 40 preserves the original Llama layer-20 relative-depth choice "
            "for a 64-layer model. Layers 32 and 48 are prespecified midpoint and "
            "three-quarter-depth checks. No layer is selected from observed AUROC."
        ),
        "fit_status_counts": _counts(rows, "fit_status"),
        "behavioral_outcome_counts": _counts(
            _unique_rows(rows, ("trajectory_id",)), "behavioral_outcome"
        ),
    }


def write_per_step_probe_scores(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write stable JSON Lines for downstream temporal path analysis."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _fit_fold_probes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    pair_ids_train: np.ndarray,
    X_test: np.ndarray,
    *,
    random_state: int,
    max_iter: int,
) -> dict[str, tuple[np.ndarray, str]]:
    goldowsky = make_probe_pipeline(
        max_iter=max_iter,
        random_state=random_state,
    )
    goldowsky.fit(X_train, y_train)
    goldowsky_probabilities = goldowsky.predict_proba(X_test)[:, 1]

    pair_differences = _contrast_pair_differences(
        X_train,
        y_train,
        pair_ids_train,
    )
    if float(np.linalg.norm(pair_differences)) == 0.0:
        lat_probabilities = np.full(len(X_test), float(np.mean(y_train)))
        lat_status = "constant_prior_identical_contrast_pairs"
    else:
        lat = ContrastPairLATProbe(random_state=random_state).fit(
            X_train,
            y_train,
            pair_ids_train,
        )
        lat_probabilities = lat.predict_proba(X_test)[:, 1]
        lat_status = "fitted"

    return {
        PROBE_NAMES[0]: (goldowsky_probabilities, "fitted"),
        PROBE_NAMES[1]: (lat_probabilities, lat_status),
    }


def preprocess_activation_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    train_groups: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fold-local activation transform without using held-out values.

    ``train_domain_mean_residualized`` subtracts each observed training
    domain's mean from its own training rows.  The held-out domain is unseen,
    so its rows receive only the training-fold grand-mean fallback.  Neither a
    held-out activation nor a held-out label contributes to either transform.
    """

    if method not in ACTIVATION_PREPROCESSING_METHODS:
        raise ValueError(
            "method must be one of "
            f"{sorted(ACTIVATION_PREPROCESSING_METHODS)}, got {method!r}."
        )
    train = np.asarray(X_train, dtype=np.float32)
    test = np.asarray(X_test, dtype=np.float32)
    groups = np.asarray(train_groups)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("Training and held-out activations must be aligned matrices.")
    if len(groups) != len(train):
        raise ValueError("train_groups must align with the training activations.")
    if method == "none":
        return train, test
    if not len(train):
        raise ValueError("Residualization requires at least one training activation.")

    domain_means = {
        group: train[groups == group].mean(axis=0, dtype=np.float64)
        for group in sorted(set(groups.tolist()))
    }
    residualized_train = np.stack(
        [row - domain_means[group] for row, group in zip(train, groups)]
    ).astype(np.float32, copy=False)
    train_grand_mean = train.mean(axis=0, dtype=np.float64)
    residualized_test = (test - train_grand_mean).astype(np.float32, copy=False)
    return residualized_train, residualized_test


def _contrast_pair_differences(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
) -> np.ndarray:
    """Return positive-minus-negative differences for complete matched pairs."""

    differences = []
    for pair_id in sorted(set(pair_ids.tolist())):
        pair_mask = pair_ids == pair_id
        pair_labels = y[pair_mask]
        if int(pair_mask.sum()) != 2 or sorted(pair_labels.tolist()) != [0, 1]:
            raise ValueError(
                f"Contrast pair {pair_id!r} must contain one clean and one injected row."
            )
        pair_activations = X[pair_mask]
        differences.append(
            pair_activations[pair_labels == 1][0]
            - pair_activations[pair_labels == 0][0]
        )
    return np.asarray(differences, dtype=np.float32)


def _score_row(
    metadata: dict[str, Any],
    *,
    score: float,
    probe_name: str,
    layer: int,
    label_target: str,
    held_out_group: str,
    fit_status: str,
    design: MatchGroupDesign,
    activation_preprocessing: str,
) -> dict[str, Any]:
    outcome = str(metadata["outcome_class"])
    if outcome not in BEHAVIORAL_OUTCOMES:
        raise ValueError(
            f"Trajectory {metadata['trajectory_id']!r} has unsupported outcome {outcome!r}."
        )
    action_label = metadata.get("labels", {}).get("unsafe_action_executed")
    if outcome == "indeterminate":
        action_fired = None
    elif action_label in (0, 1, False, True):
        action_fired = bool(action_label)
    else:
        raise ValueError(
            f"Trajectory {metadata['trajectory_id']!r} has no binary "
            "unsafe_action_executed label."
        )
    label_value = metadata.get("labels", {}).get(label_target)
    latent_status = (
        "candidate"
        if outcome in {"propagated_but_not_executed", "attempted_but_blocked"}
        else "not_candidate"
    )
    hop_index = int(metadata["hop_index"])
    row = {
        "schema_version": PER_STEP_SCORE_SCHEMA,
        "artifact_kind": "held_out_probe_score",
        "claim_scope": "construction_label_diagnostic",
        "analysis_tier": normalize_index_analysis_tier(
            metadata.get("analysis_tier") or "unclassified"
        ),
        "trajectory_id": str(metadata["trajectory_id"]),
        "match_group_id": str(metadata["match_group_id"]),
        "domain_id": str(metadata["domain_id"]),
        "wording_id": design.wording_id,
        "condition": str(metadata["treatment"]),
        "injection_present": bool(metadata["labels"]["injection_present"]),
        "hop_mode": str(metadata["delegation_depth"]),
        "agent_id": str(metadata["agent_id"]),
        "agent_role": str(metadata["agent_role"]),
        "hop_index": hop_index,
        "distance_from_injection": hop_index - 1,
        "anchor_agent_id": "worker_1",
        "anchor_hop_index": 1,
        "model": str(metadata["model_name"]),
        "model_revision": str(metadata["model_revision"]),
        "thinking_mode": str(metadata["thinking_mode"]),
        "checkpoint": str(metadata["checkpoint"]),
        "checkpoint_forward_scope": metadata.get("checkpoint_forward_scope"),
        "layer": layer,
        "probe_name": probe_name,
        "score": score,
        "label": int(label_value) if label_value in (0, 1) else None,
        "label_target": label_target,
        "behavioral_outcome": outcome,
        "action_fired": action_fired,
        "latent_compromise_status": latent_status,
        "seed": design.seed,
        "evaluation_method": EVALUATION_METHOD,
        "held_out_match_group_id": held_out_group,
        "fit_status": fit_status,
    }
    if activation_preprocessing != "none":
        row["activation_preprocessing"] = activation_preprocessing
    return row


def _validate_index_design(
    rows: Sequence[dict[str, Any]],
    match_group_designs: dict[str, MatchGroupDesign],
    *,
    label_target: str,
) -> None:
    analysis_tiers = {
        normalize_index_analysis_tier(row.get("analysis_tier") or "unclassified")
        for row in rows
    }
    if len(analysis_tiers) != 1:
        raise ValueError("Activation-index rows must contain one analysis tier.")
    observed_groups = {str(row.get("match_group_id")) for row in rows}
    missing_designs = sorted(observed_groups - set(match_group_designs))
    if missing_designs:
        raise ValueError(
            f"Manifest metadata is missing match groups: {missing_designs}"
        )
    for index, row in enumerate(rows):
        if row.get("labels", {}).get(label_target) not in (0, 1):
            raise ValueError(
                f"Activation-index row {index} has no binary {label_target!r} label."
            )
        if row.get("thinking_mode") not in {"off", "on"}:
            raise ValueError(f"Activation-index row {index} has invalid thinking_mode.")
        if row.get("delegation_depth") not in {"2-hop", "3-hop"}:
            raise ValueError(
                f"Activation-index row {index} has invalid delegation_depth."
            )
        if row.get("treatment") not in {"clean", "injected"}:
            raise ValueError(f"Activation-index row {index} has invalid treatment.")


def _counts(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _unique_rows(
    rows: Sequence[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(tuple(row[field] for field in fields), row)
    return list(unique.values())


def _index_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["match_group_id"]),
        str(row["delegation_depth"]),
        str(row["treatment"]),
        str(row["trajectory_id"]),
        int(row["hop_index"]),
    )


def _score_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["probe_name"]),
        str(row["thinking_mode"]),
        int(row["layer"]),
        str(row["match_group_id"]),
        str(row["hop_mode"]),
        str(row["condition"]),
        str(row["trajectory_id"]),
        int(row["hop_index"]),
    )
