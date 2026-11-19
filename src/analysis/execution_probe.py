"""Executed-vs-propagated probe for the lexical-sandbox activations (paper §5.6).

The question (paper Eq. 2): among trajectories in which the injected objective has
already *propagated*, can the executor-stage internal state predict which culminate
in an unauthorized executed tool call?

    Z = 1[Y = executed],   Y != resisted.

Both classes carried the injected objective downstream, so injection presence alone
cannot define the target. We train a group-safe logistic probe on the executor's
last-input-token residual stream (captured by
``scripts/02_model_execution/09_run_paraphrase_dose.py``) to separate ``executed``
from ``propagated``, report leave-one-domain-out AUROC and the executed-vs-propagated
Cohen's d, and — mirroring the paper — compare a grid-only logistic model of Z to one
that adds the standardized probe margin, so the reported quantity is the probe's value
*beyond* the experimental cell a trajectory came from.

Inputs are the two sidecars the runner writes next to each dose-response JSON:
``..._activations.npz`` (``X`` of shape (N, n_layers, hidden); ``layers``) and
``..._activations_manifest.json`` (one row per trajectory, with ``outcome`` and the
grid factors). Several runs can be pooled by passing multiple file pairs.

These are probe scores from representation activations, not new model generations and
not inferred behavioral outcomes; every out-of-fold score is produced by a model fitted
only on other domains.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler


POSITIVE_OUTCOMES = frozenset({"executed"})
# Y != resisted and not executed: reached-the-executor outcomes that did not fire.
NEGATIVE_OUTCOMES = frozenset({"propagated_but_not_executed", "attempted_but_blocked"})
EXCLUDED_OUTCOMES = frozenset({"resisted", "clean"})

GRID_FACTORS = ("domain", "style_key", "position", "condition", "thinking_mode", "level")


@dataclass(frozen=True)
class ProbeDataset:
    """Executor-stage activations and labels, restricted to Y != resisted."""

    X: np.ndarray            # (N, hidden) for the chosen layer
    y: np.ndarray            # 1 = executed, 0 = propagated
    groups: np.ndarray       # group id per row (default: domain) for group-safe CV
    rows: list[dict]         # the surviving manifest rows, aligned to X/y
    layer: int
    n_excluded_resisted: int


@dataclass
class ProbeResult:
    layer: int
    n: int
    n_executed: int
    n_propagated: int
    n_groups: int
    cv_scheme: str
    auroc: float | None
    cohens_d: float | None
    # §5.6 incremental: logistic model of Z from grid factors, vs grid + probe margin.
    grid_only_auroc: float | None = None
    grid_only_brier: float | None = None
    grid_plus_probe_auroc: float | None = None
    grid_plus_probe_brier: float | None = None
    auroc_gain: float | None = None
    brier_reduction: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Executed-vs-propagated probe (layer {self.layer}, "
            f"n={self.n}: {self.n_executed} executed / {self.n_propagated} propagated, "
            f"{self.n_groups} groups, CV={self.cv_scheme})",
        ]
        if self.auroc is None:
            lines.append("  AUROC: n/a (" + "; ".join(self.notes) + ")")
            return "\n".join(lines)
        lines.append(
            f"  probe AUROC {self.auroc:.3f} | executed-vs-propagated Cohen's d "
            f"{self.cohens_d:+.2f}"
        )
        if self.grid_only_auroc is not None:
            lines.append(
                f"  grid-only AUROC {self.grid_only_auroc:.3f} (Brier "
                f"{self.grid_only_brier:.3f}) -> +probe AUROC "
                f"{self.grid_plus_probe_auroc:.3f} (Brier {self.grid_plus_probe_brier:.3f}) "
                f"| gain {self.auroc_gain:+.3f}, Brier {self.brier_reduction:+.3f}"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def load_probe_dataset(
    file_pairs: Sequence[tuple[str | Path, str | Path]],
    layer: int | None = None,
    group_by: str = "domain",
) -> ProbeDataset:
    """Load and pool one or more (activations.npz, manifest.json) pairs.

    ``layer`` selects which captured layer to probe; default is the first captured
    layer (the profile's primary/reference layer). Rows whose outcome is resisted or
    clean are dropped (Y != resisted); remaining rows are labeled executed=1 else 0.
    """

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    kept_rows: list[dict] = []
    n_excluded = 0
    chosen_layer: int | None = None

    for npz_path, manifest_path in file_pairs:
        with np.load(npz_path) as data:
            X = np.asarray(data["X"], dtype=np.float32)  # (N, n_layers, hidden)
            layers = [int(v) for v in data["layers"]]
        manifest = json.loads(Path(manifest_path).read_text())
        rows = manifest["rows"]
        if X.shape[0] != len(rows):
            raise ValueError(
                f"{npz_path}: X has {X.shape[0]} rows but manifest has {len(rows)}"
            )
        want = layers[0] if layer is None else layer
        if want not in layers:
            raise ValueError(f"layer {want} not in captured layers {layers} for {npz_path}")
        if chosen_layer is None:
            chosen_layer = want
        elif chosen_layer != want:
            raise ValueError("pooled runs disagree on the chosen layer")
        li = layers.index(want)

        for i, row in enumerate(rows):
            outcome = row.get("outcome")
            if outcome in EXCLUDED_OUTCOMES:
                n_excluded += 1
                continue
            if outcome in POSITIVE_OUTCOMES:
                label = 1
            elif outcome in NEGATIVE_OUTCOMES:
                label = 0
            else:
                n_excluded += 1
                continue
            X_parts.append(X[i, li, :])
            y_parts.append(label)
            group_parts.append(str(row.get(group_by, "all")))
            kept_rows.append(row)

    if not X_parts:
        return ProbeDataset(
            X=np.empty((0, 0)), y=np.empty((0,), dtype=int),
            groups=np.empty((0,), dtype=object), rows=[], layer=chosen_layer or -1,
            n_excluded_resisted=n_excluded,
        )
    return ProbeDataset(
        X=np.vstack(X_parts),
        y=np.asarray(y_parts, dtype=int),
        groups=np.asarray(group_parts, dtype=object),
        rows=kept_rows,
        layer=chosen_layer,
        n_excluded_resisted=n_excluded,
    )


def _cohens_d(scores: np.ndarray, y: np.ndarray) -> float:
    a, b = scores[y == 1], scores[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


def _oof_probe_margin(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, random_state: int,
) -> tuple[np.ndarray | None, str]:
    """Pooled out-of-fold standardized logit of the executed-vs-propagated probe.

    Leave-one-group-out when there are >=2 groups each usable; otherwise stratified
    5-fold. Returns None if no fold can train on both classes.
    """

    n_groups = len(set(groups.tolist()))
    use_logo = n_groups >= 2
    splitter = LeaveOneGroupOut() if use_logo else StratifiedKFold(
        n_splits=min(5, int(np.bincount(y).min())) or 2, shuffle=True,
        random_state=random_state,
    )
    scheme = "leave-one-group-out" if use_logo else "stratified-5fold"
    split_iter = splitter.split(X, y, groups) if use_logo else splitter.split(X, y)

    margins = np.full(len(y), np.nan, dtype=float)
    trained_any = False
    for train_idx, test_idx in split_iter:
        if len(set(y[train_idx].tolist())) < 2:
            continue  # fold's train side has only one class; cannot fit
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=C, max_iter=2000, random_state=random_state)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        margins[test_idx] = clf.decision_function(scaler.transform(X[test_idx]))
        trained_any = True
    if not trained_any or np.isnan(margins).all():
        return None, scheme
    return margins, scheme


def _onehot(rows: list[dict], factors: Sequence[str]) -> np.ndarray:
    """Dense one-hot of the grid factors (drop-first per factor to avoid collinearity)."""

    columns: list[np.ndarray] = []
    for factor in factors:
        values = [str(r.get(factor)) for r in rows]
        levels = sorted(set(values))
        if len(levels) < 2:
            continue  # constant factor carries no information
        for lvl in levels[1:]:  # drop first level as the reference
            columns.append(np.asarray([1.0 if v == lvl else 0.0 for v in values]))
    if not columns:
        return np.zeros((len(rows), 0))
    return np.vstack(columns).T


def probe_executed_vs_propagated(
    dataset: ProbeDataset,
    C: float = 1.0,
    random_state: int = 0,
    grid_factors: Sequence[str] = GRID_FACTORS,
) -> ProbeResult:
    """Group-safe executed-vs-propagated probe + the §5.6 grid-vs-probe comparison."""

    y = dataset.y
    n_exec = int((y == 1).sum())
    n_prop = int((y == 0).sum())
    n_groups = len(set(dataset.groups.tolist())) if len(y) else 0
    result = ProbeResult(
        layer=dataset.layer, n=len(y), n_executed=n_exec, n_propagated=n_prop,
        n_groups=n_groups, cv_scheme="n/a", auroc=None, cohens_d=None,
    )
    if n_exec < 2 or n_prop < 2:
        result.notes.append(
            f"need >=2 of each class; have {n_exec} executed / {n_prop} propagated"
        )
        return result

    margins, scheme = _oof_probe_margin(dataset.X, y, dataset.groups, C, random_state)
    result.cv_scheme = scheme
    if margins is None:
        result.notes.append("no CV fold could train on both classes (group confound)")
        return result

    mask = ~np.isnan(margins)
    if len(set(y[mask].tolist())) < 2:
        result.notes.append("out-of-fold predictions cover only one class")
        return result
    result.auroc = float(roc_auc_score(y[mask], margins[mask]))
    if min(n_exec, n_prop) < 5:
        result.notes.append(
            f"low power: only {min(n_exec, n_prop)} in the smaller class — AUROC is noisy"
        )
    std = margins[mask].std()
    z_margin = (margins - margins[mask].mean()) / (std if std else 1.0)
    result.cohens_d = _cohens_d(z_margin[mask], y[mask])

    # §5.6 incremental: logistic Z ~ grid, vs Z ~ grid + standardized probe margin.
    grid = _onehot([dataset.rows[i] for i in range(len(y))], grid_factors)
    gm = grid[mask]
    ym = y[mask]
    try:
        grid_only = LogisticRegression(C=1e6, max_iter=5000).fit(gm, ym)
        p_grid = grid_only.predict_proba(gm)[:, 1]
        feats = np.hstack([gm, z_margin[mask].reshape(-1, 1)])
        grid_probe = LogisticRegression(C=1e6, max_iter=5000).fit(feats, ym)
        p_both = grid_probe.predict_proba(feats)[:, 1]
        result.grid_only_auroc = float(roc_auc_score(ym, p_grid))
        result.grid_only_brier = float(brier_score_loss(ym, p_grid))
        result.grid_plus_probe_auroc = float(roc_auc_score(ym, p_both))
        result.grid_plus_probe_brier = float(brier_score_loss(ym, p_both))
        result.auroc_gain = result.grid_plus_probe_auroc - result.grid_only_auroc
        result.brier_reduction = result.grid_plus_probe_brier - result.grid_only_brier
        result.notes.append(
            "grid/grid+probe fit is in-sample (describes incremental fit, not held-out "
            "generalization); the probe margin is out-of-fold"
        )
    except Exception as exc:  # pragma: no cover - degenerate grids
        result.notes.append(f"grid-vs-probe comparison skipped: {exc}")
    return result


def run_from_files(
    file_pairs: Sequence[tuple[str | Path, str | Path]],
    layer: int | None = None,
    group_by: str = "domain",
    C: float = 1.0,
) -> ProbeResult:
    dataset = load_probe_dataset(file_pairs, layer=layer, group_by=group_by)
    return probe_executed_vs_propagated(dataset, C=C)


__all__ = [
    "ProbeDataset",
    "ProbeResult",
    "load_probe_dataset",
    "probe_executed_vs_propagated",
    "run_from_files",
    "POSITIVE_OUTCOMES",
    "NEGATIVE_OUTCOMES",
    "EXCLUDED_OUTCOMES",
    "GRID_FACTORS",
]
