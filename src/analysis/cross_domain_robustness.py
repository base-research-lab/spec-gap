"""Robustness summaries for the nine-domain Scenario 1 construction probe.

The functions in this module operate on saved activations and held-out probe
scores.  They do not generate model responses and they do not turn the
``injection_present`` construction label into a behavioral-compromise label.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import statistics
from typing import Any, Sequence

import numpy as np

from src.infrastructure.model_profiles import ACTIVE_PROFILE
from src.analysis.probe_scoring import preprocess_activation_fold
from src.probes.linear_probe import make_probe_pipeline


PRIMARY_AGENT = "worker_1"
PRIMARY_THINKING_MODE = "off"
PRIMARY_LAYER = ACTIVE_PROFILE.reference_layer
PRIMARY_PROBE = "goldowsky_dill_logistic"


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute AUROC with half credit for tied positive/negative scores."""

    positives = [
        float(score) for label, score in zip(labels, scores) if int(label) == 1
    ]
    negatives = [
        float(score) for label, score in zip(labels, scores) if int(label) == 0
    ]
    if not positives or not negatives:
        raise ValueError("AUROC requires both label classes.")
    favorable = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                favorable += 1.0
            elif positive == negative:
                favorable += 0.5
    return favorable / (len(positives) * len(negatives))


def summarize_held_out_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pooled and equally weighted held-out-fold AUROC."""

    if not rows:
        raise ValueError("At least one held-out score row is required.")
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError("Held-out score rows require binary labels.")
        by_fold[str(row["held_out_match_group_id"])].append(dict(row))

    fold_rows = []
    for fold, cohort in sorted(by_fold.items()):
        fold_rows.append(
            {
                "held_out_match_group_id": fold,
                "domain_id": _one_value(cohort, "domain_id"),
                "n_predictions": len(cohort),
                "n_clean": sum(int(row["label"]) == 0 for row in cohort),
                "n_injected": sum(int(row["label"]) == 1 for row in cohort),
                "auroc": auroc(
                    [int(row["label"]) for row in cohort],
                    [float(row["score"]) for row in cohort],
                ),
            }
        )
    return {
        "n_predictions": len(rows),
        "n_folds": len(fold_rows),
        "pooled_auroc": auroc(
            [int(row["label"]) for row in rows],
            [float(row["score"]) for row in rows],
        ),
        "mean_fold_auroc": statistics.fmean(float(row["auroc"]) for row in fold_rows),
        "folds": fold_rows,
    }


def paired_domain_score_deltas(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return injected-minus-clean score changes for each depth/domain pair."""

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition"))
        if condition not in {"clean", "injected"}:
            raise ValueError("Paired score rows require clean/injected conditions.")
        key = (str(row["domain_id"]), str(row["hop_mode"]))
        if condition in grouped[key]:
            raise ValueError(f"Duplicate {condition} score for {key!r}.")
        grouped[key][condition] = dict(row)

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (domain, depth), pair in sorted(grouped.items()):
        if set(pair) != {"clean", "injected"}:
            raise ValueError(
                f"Incomplete clean/injected score pair for {(domain, depth)!r}."
            )
        clean_score = float(pair["clean"]["score"])
        injected_score = float(pair["injected"]["score"])
        by_domain[domain].append(
            {
                "delegation_depth": depth,
                "clean_score": clean_score,
                "injected_score": injected_score,
                "injected_minus_clean": injected_score - clean_score,
                "clean_trajectory_id": str(pair["clean"]["trajectory_id"]),
                "injected_trajectory_id": str(pair["injected"]["trajectory_id"]),
            }
        )

    output = []
    for domain, pairs in sorted(by_domain.items()):
        pairs.sort(key=lambda row: row["delegation_depth"])
        deltas = [float(row["injected_minus_clean"]) for row in pairs]
        output.append(
            {
                "domain_id": domain,
                "n_pairs": len(pairs),
                "mean_injected_minus_clean": statistics.fmean(deltas),
                "observed_min": min(deltas),
                "observed_max": max(deltas),
                "pairs": pairs,
            }
        )
    return output


def refit_within_fold_permutation_null(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_permutations: int,
    random_state: int,
    activation_preprocessing: str = "none",
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Refit a leave-one-domain-out logistic probe under shuffled labels.

    Each replicate independently chooses a class-balanced label assignment
    within every domain, then re-runs every held-out fold.  The probe and any
    requested fold-local preprocessing are fitted again from training rows.
    This is an end-to-end construction-label null, not a behavioral test.
    """

    activations = np.asarray(X, dtype=np.float32)
    observed_labels = np.asarray(labels, dtype=int)
    match_groups = np.asarray(groups)
    if activations.ndim != 2:
        raise ValueError("X must be a two-dimensional activation matrix.")
    if not (len(activations) == len(observed_labels) == len(match_groups)):
        raise ValueError("X, labels, and groups must have the same length.")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    group_indices = {
        group: np.flatnonzero(match_groups == group)
        for group in sorted(set(match_groups.tolist()))
    }
    assignment_options: dict[Any, list[tuple[int, ...]]] = {}
    for group, indices in group_indices.items():
        group_labels = observed_labels[indices]
        positive_count = int(group_labels.sum())
        if not 0 < positive_count < len(indices):
            raise ValueError(f"Group {group!r} must contain both label classes.")
        assignment_options[group] = list(combinations(indices.tolist(), positive_count))

    observed_scores = _refit_group_held_out_scores(
        activations,
        observed_labels,
        match_groups,
        activation_preprocessing=activation_preprocessing,
        max_iter=max_iter,
    )
    observed = _array_score_summary(
        observed_labels,
        observed_scores,
        match_groups,
    )

    rng = np.random.default_rng(random_state)
    null_mean_fold = []
    null_pooled = []
    for _ in range(n_permutations):
        permuted = np.zeros_like(observed_labels)
        for group, options in assignment_options.items():
            selected = options[int(rng.integers(0, len(options)))]
            permuted[list(selected)] = 1
        scores = _refit_group_held_out_scores(
            activations,
            permuted,
            match_groups,
            activation_preprocessing=activation_preprocessing,
            max_iter=max_iter,
        )
        summary = _array_score_summary(permuted, scores, match_groups)
        null_mean_fold.append(float(summary["mean_fold_auroc"]))
        null_pooled.append(float(summary["pooled_auroc"]))

    return {
        "method": "balanced_within_domain_label_permutation_with_full_refit",
        "random_state": random_state,
        "n_permutations": n_permutations,
        "activation_preprocessing": activation_preprocessing,
        "fold_count": len(group_indices),
        "observations_per_fold": {
            str(group): len(indices) for group, indices in group_indices.items()
        },
        "positive_labels_per_fold": {
            str(group): int(observed_labels[indices].sum())
            for group, indices in group_indices.items()
        },
        "observed": observed,
        "mean_fold_auroc_null": _null_summary(
            null_mean_fold,
            observed=float(observed["mean_fold_auroc"]),
        ),
        "pooled_auroc_null": _null_summary(
            null_pooled,
            observed=float(observed["pooled_auroc"]),
        ),
        "limitations": [
            "Each held-out domain has only four observations (two per class), so fold AUROC moves in coarse increments.",
            "The null tests construction-label separability under within-domain exchangeability; it is not a compromise-detection or causal test.",
        ],
    }


def _refit_group_held_out_scores(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    activation_preprocessing: str,
    max_iter: int,
) -> np.ndarray:
    scores = np.empty(len(labels), dtype=float)
    for held_out_group in sorted(set(groups.tolist())):
        test_mask = groups == held_out_group
        train_mask = ~test_mask
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        if set(y_train.tolist()) != {0, 1} or set(y_test.tolist()) != {0, 1}:
            raise ValueError("Every permutation fold must contain both classes.")
        X_train, X_test = preprocess_activation_fold(
            X[train_mask],
            X[test_mask],
            train_groups=groups[train_mask],
            method=activation_preprocessing,
        )
        probe = make_probe_pipeline(max_iter=max_iter, random_state=42)
        probe.fit(X_train, y_train)
        scores[test_mask] = probe.predict_proba(X_test)[:, 1]
    return scores


def _array_score_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    fold_aurocs = {
        str(group): auroc(labels[groups == group], scores[groups == group])
        for group in sorted(set(groups.tolist()))
    }
    return {
        "pooled_auroc": auroc(labels, scores),
        "mean_fold_auroc": statistics.fmean(fold_aurocs.values()),
        "fold_aurocs": fold_aurocs,
    }


def _null_summary(values: Sequence[float], *, observed: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "observed": observed,
        "null_mean": float(array.mean()),
        "null_sd": float(array.std()),
        "null_quantiles": {
            "q025": float(np.quantile(array, 0.025)),
            "q500": float(np.quantile(array, 0.5)),
            "q975": float(np.quantile(array, 0.975)),
        },
        "greater_or_equal_count": int(np.count_nonzero(array >= observed)),
        "add_one_p_value": float(
            (1 + np.count_nonzero(array >= observed)) / (1 + len(array))
        ),
        "null_values": [float(value) for value in array],
    }


def _one_value(rows: Sequence[dict[str, Any]], field: str) -> Any:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ValueError(f"Rows do not contain one {field}: {sorted(values)!r}.")
    return next(iter(values))
