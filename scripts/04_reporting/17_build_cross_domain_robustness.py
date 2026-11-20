#!/usr/bin/env python3
"""Build robustness checks for the nine-domain construction-signal analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.cross_domain_robustness import (  # noqa: E402
    PRIMARY_AGENT,
    PRIMARY_LAYER,
    PRIMARY_PROBE,
    PRIMARY_THINKING_MODE,
    paired_domain_score_deltas,
    refit_within_fold_permutation_null,
    summarize_held_out_scores,
)
from src.analysis.depth_degradation import (  # noqa: E402
    load_prediction_jsonl,
    prediction_analysis_tier,
)
from src.analysis.figure_export import save_reproducible_figure  # noqa: E402
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    validate_paper_analysis_inputs,
)
from src.analysis.probe_scoring import (  # noqa: E402
    generate_per_step_probe_scores,
    load_match_group_designs,
)
from src.extraction.saved_activations import (  # noqa: E402
    activation_index_analysis_tier,
    load_activation_index,
    load_probe_activation_batch,
)


DOMAIN_LABELS = {
    "aihc": "AIHC",
    "convex": "Convex",
    "fin": "Finance",
    "kg": "Knowledge Graphs",
    "macro": "Macro",
    "neuro": "Neuro",
    "petro": "Petroleum",
    "policy": "Policy",
    "telecom": "Telecom",
}
MATCH_GROUP_TO_DOMAIN = {
    "aihc": "aihc",
    "convex_open_access_v3": "convex",
    "fin": "fin",
    "kg": "kg",
    "macro": "macro",
    "neuro": "neuro",
    "petro": "petro",
    "policy": "policy",
    "telecom": "telecom",
}
STYLE_BY_GROUP = {
    "aihc": "plain_text",
    "convex_open_access_v3": "chat_special_tokens_and_explicit_tool_syntax",
    "fin": "plain_text",
    "kg": "chat_special_tokens_and_explicit_tool_syntax",
    "macro": "think_tag_wrapped_text",
    "neuro": "plain_text",
    "petro": "plain_text",
    "policy": "plain_text",
    "telecom": "plain_text",
}
EXPOSURE_BY_GROUP = {
    "aihc": "natural_only",
    "convex_open_access_v3": "natural_only",
    "fin": "natural_only",
    "kg": "natural_only",
    "macro": "natural_only",
    "neuro": "natural_only",
    "petro": "natural_only",
    "policy": "require_clean_anchor_position_adjusted",
    "telecom": "require_clean_anchor_position_adjusted",
}
COHORTS = {
    "all_nine_domains": tuple(MATCH_GROUP_TO_DOMAIN),
    "remove_kg_and_convex": (
        "aihc",
        "fin",
        "macro",
        "neuro",
        "petro",
        "policy",
        "telecom",
    ),
    "plain_text_six_domains": (
        "aihc",
        "fin",
        "neuro",
        "petro",
        "policy",
        "telecom",
    ),
    "special_token_tool_syntax_two_domains": (
        "convex_open_access_v3",
        "kg",
    ),
    "natural_selection_seven_domains": (
        "aihc",
        "convex_open_access_v3",
        "fin",
        "kg",
        "macro",
        "neuro",
        "petro",
    ),
    "required_anchor_two_domains": ("policy", "telecom"),
    "think_tag_single_domain": ("macro",),
}


def _display_path(path: Path) -> str:
    """Prefer a repository-relative source path when the file is in-tree."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Re-fit layer-{PRIMARY_LAYER} Worker 1 probes and build style, exposure, paired-"
            "delta, permutation, and train-only residualization checks."
        )
    )
    parser.add_argument("--activation-index", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--design-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--random-state", type=int, default=20260806)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
    )
    args = parser.parse_args()

    index_rows = load_activation_index(args.activation_index)
    all_score_rows = load_prediction_jsonl(args.scores)
    paper_policy = load_paper_input_policy(args.paper_input_policy)
    index_selection = validate_paper_analysis_inputs(index_rows, paper_policy)
    score_selection = validate_paper_analysis_inputs(all_score_rows, paper_policy)
    analysis_tiers = {
        activation_index_analysis_tier(index_rows),
        prediction_analysis_tier(all_score_rows),
    }
    if len(analysis_tiers) != 1:
        raise ValueError("Robustness inputs must contain one matching analysis tier.")
    if (
        index_selection["selected_trajectory_sha256"]
        != score_selection["selected_trajectory_sha256"]
    ):
        raise ValueError(
            "Robustness inputs do not contain the same paper-input cohort."
        )
    analysis_tier = next(iter(analysis_tiers))
    designs = load_match_group_designs(args.design_manifest)
    primary_index = sorted(
        (
            row
            for row in index_rows
            if row["thinking_mode"] == PRIMARY_THINKING_MODE
            and row["agent_id"] == PRIMARY_AGENT
            and row["checkpoint"] == "last_input_token"
        ),
        key=_index_sort_key,
    )
    primary_saved_scores = sorted(
        (
            row
            for row in all_score_rows
            if row["thinking_mode"] == PRIMARY_THINKING_MODE
            and row["agent_id"] == PRIMARY_AGENT
            and int(row["layer"]) == PRIMARY_LAYER
        ),
        key=_score_sort_key,
    )
    expected_groups = set(MATCH_GROUP_TO_DOMAIN)
    observed_groups = {str(row["match_group_id"]) for row in primary_index}
    if observed_groups != expected_groups:
        raise ValueError(
            "Primary activation cohort does not contain the expected nine groups: "
            f"{sorted(observed_groups)}."
        )

    regenerated = generate_per_step_probe_scores(
        primary_index,
        match_group_designs=designs,
        layers=[PRIMARY_LAYER],
        verify_checksums=True,
    )
    if regenerated != primary_saved_scores:
        raise ValueError(
            f"Saved Worker 1 layer-{PRIMARY_LAYER} scores do not match an exact activation re-fit."
        )

    cohort_results, compact_score_rows = _build_cohort_results(
        primary_index,
        regenerated,
        designs,
    )
    residualized_scores = generate_per_step_probe_scores(
        primary_index,
        match_group_designs=designs,
        layers=[PRIMARY_LAYER],
        verify_checksums=True,
        activation_preprocessing="train_domain_mean_residualized",
    )
    residualization = {
        probe: {
            "before": summarize_held_out_scores(
                [row for row in regenerated if row["probe_name"] == probe]
            ),
            "after": summarize_held_out_scores(
                [row for row in residualized_scores if row["probe_name"] == probe]
            ),
        }
        for probe in sorted({row["probe_name"] for row in regenerated})
    }

    goldowsky_rows = [row for row in regenerated if row["probe_name"] == PRIMARY_PROBE]
    paired_deltas = paired_domain_score_deltas(goldowsky_rows)
    activation_batch = load_probe_activation_batch(
        primary_index,
        label_target="injection_present",
        layers=[PRIMARY_LAYER],
        verify_checksums=True,
    )
    permutation_null = refit_within_fold_permutation_null(
        activation_batch.activations[PRIMARY_LAYER],
        activation_batch.labels,
        groups=_groups_from_metadata(activation_batch.metadata),
        n_permutations=args.permutations,
        random_state=args.random_state,
    )

    outcome_counts = _trajectory_outcome_counts(index_rows)
    sample_sizes = _agent_sample_sizes(all_score_rows)
    design_table = _design_table()
    source_hashes = {
        "activation_index": {
            "path": _display_path(args.activation_index),
            "sha256": _sha256(args.activation_index),
            "rows": len(index_rows),
        },
        "saved_probe_scores": {
            "path": _display_path(args.scores),
            "sha256": _sha256(args.scores),
            "rows": len(all_score_rows),
        },
        "probe_design": {
            "path": _display_path(args.design_manifest),
            "sha256": _sha256(args.design_manifest),
        },
        "paper_input_policy": {
            "path": _display_path(args.paper_input_policy),
            "sha256": _sha256(args.paper_input_policy),
        },
        "activation_artifacts": {
            "files": len({row["local_path"] for row in index_rows}),
            "all_checksums_verified": True,
        },
    }
    artifact = {
        "schema_version": "spec_gap.cross_domain_robustness.v2",
        "analysis_tier": analysis_tier,
        "paper_input_selection": index_selection,
        "created_at": "2026-08-10",
        "analysis_scope": {
            "label": "injection_present",
            "meaning": "presence of injected prompt construction/tokens",
            "not_measured": "behavioral compromise detection",
            "compromise_auroc_estimable": False,
            "reason": (
                "All 36 injected trajectories were labeled resisted, so the "
                "sample has no positive compromise outcome."
            ),
            "new_model_generations": 0,
        },
        "primary_analysis": {
            "agent_id": PRIMARY_AGENT,
            "thinking_mode": PRIMARY_THINKING_MODE,
            "layer": PRIMARY_LAYER,
            "probe": PRIMARY_PROBE,
        },
        "source_hashes": source_hashes,
        "design_covariates": design_table,
        "trajectory_outcomes": outcome_counts,
        "agent_sample_sizes": sample_sizes,
        "cohort_analyses": cohort_results,
        "paired_injected_minus_clean_scores": paired_deltas,
        "permutation_null": permutation_null,
        "train_fold_only_domain_mean_residualization": {
            "method": (
                "Training rows are centered by their training-domain means. "
                "The unseen held-out domain receives only the training-fold "
                "grand-mean fallback; no held-out activation or label is used "
                "to fit the transform."
            ),
            "results": residualization,
        },
        "score_evidence": {
            "cohort_refits": compact_score_rows,
            "all_domain_train_residualized": residualized_scores,
        },
        "interpretation": [
            "The 0.889 headline is mean held-out-domain construction-label AUROC, not pooled AUROC and not compromise-detection AUROC.",
            "AIHC is the only zero-AUROC primary fold; the other eight primary folds are 1.0.",
            "Style and exposure cohorts are small, so changes across ablations are sensitivity checks rather than stable mechanism estimates.",
            "A future natural-text attack redesign belongs to the research group and is not part of this cleanup artifact.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cross_domain_robustness.json"
    markdown_path = args.output_dir / "cross_domain_robustness.md"
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(artifact), encoding="utf-8")
    _write_tables(args.output_dir, artifact)
    figure_paths = _plot_paired_deltas(
        paired_deltas,
        output_dir=args.output_dir / "figures",
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "artifact": json_path.as_posix(),
                "summary": markdown_path.as_posix(),
                "figures": [path.as_posix() for path in figure_paths],
                "permutations": args.permutations,
                "baseline_reproduced": True,
            },
            indent=2,
        )
    )


def _build_cohort_results(
    index_rows: list[dict[str, Any]],
    baseline_scores: list[dict[str, Any]],
    designs: dict,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    for cohort_name, groups in COHORTS.items():
        selected = set(groups)
        excluded = sorted(set(MATCH_GROUP_TO_DOMAIN) - selected)
        saved_subset = [
            row for row in baseline_scores if row["held_out_match_group_id"] in selected
        ]
        fold_exclusion = {
            probe: summarize_held_out_scores(
                [row for row in saved_subset if row["probe_name"] == probe]
            )
            for probe in sorted({row["probe_name"] for row in saved_subset})
        }
        result = {
            "match_groups": list(groups),
            "domain_ids": [MATCH_GROUP_TO_DOMAIN[group] for group in groups],
            "domain_count": len(groups),
            "excluded_match_groups": excluded,
            "existing_nine_domain_fits_with_folds_filtered": fold_exclusion,
        }
        if len(groups) < 2:
            result["full_training_and_evaluation_refit"] = {
                "status": "not_estimable",
                "reason": "Leave-one-domain-out evaluation requires at least two domains.",
            }
            evidence[cohort_name] = []
        else:
            cohort_index = [
                row for row in index_rows if row["match_group_id"] in selected
            ]
            refit = generate_per_step_probe_scores(
                cohort_index,
                match_group_designs=designs,
                layers=[PRIMARY_LAYER],
                verify_checksums=True,
            )
            result["full_training_and_evaluation_refit"] = {
                probe: summarize_held_out_scores(
                    [row for row in refit if row["probe_name"] == probe]
                )
                for probe in sorted({row["probe_name"] for row in refit})
            }
            evidence[cohort_name] = refit
        results[cohort_name] = result
    return results, evidence


def _design_table() -> list[dict[str, Any]]:
    rows = []
    for match_group, domain in MATCH_GROUP_TO_DOMAIN.items():
        rows.append(
            {
                "domain_id": domain,
                "domain_label": DOMAIN_LABELS[domain],
                "match_group_id": match_group,
                "injection_style": STYLE_BY_GROUP[match_group],
                "carrier_retention_and_position": EXPOSURE_BY_GROUP[match_group],
            }
        )
    return rows


def _trajectory_outcome_counts(index_rows: list[dict[str, Any]]) -> dict[str, int]:
    trajectories: dict[str, tuple[str, str]] = {}
    for row in index_rows:
        trajectories[str(row["trajectory_id"])] = (
            str(row["treatment"]),
            str(row["outcome_class"]),
        )
    counts: dict[str, int] = {}
    for treatment, outcome in trajectories.values():
        key = f"{treatment}:{outcome}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _agent_sample_sizes(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in score_rows
        if int(row["layer"]) == PRIMARY_LAYER and row["probe_name"] == PRIMARY_PROBE
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in selected:
        key = (str(row["thinking_mode"]), str(row["agent_id"]))
        grouped.setdefault(key, []).append(row)
    return [
        {
            "thinking_mode": mode,
            "agent_id": agent,
            "predictions": len(rows),
            "positive_labels": sum(int(row["label"]) == 1 for row in rows),
            "domains": len({row["held_out_match_group_id"] for row in rows}),
        }
        for (mode, agent), rows in sorted(grouped.items())
    ]


def _write_tables(output_dir: Path, artifact: dict[str, Any]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    cohort_rows = []
    for cohort, result in artifact["cohort_analyses"].items():
        for analysis_kind, metrics_by_probe in (
            (
                "fold_exclusion_only",
                result["existing_nine_domain_fits_with_folds_filtered"],
            ),
            (
                "full_cohort_refit",
                result["full_training_and_evaluation_refit"],
            ),
        ):
            if metrics_by_probe.get("status") == "not_estimable":
                continue
            for probe, metrics in metrics_by_probe.items():
                cohort_rows.append(
                    {
                        "cohort": cohort,
                        "analysis_kind": analysis_kind,
                        "probe": probe,
                        "domain_count": result["domain_count"],
                        "prediction_count": metrics["n_predictions"],
                        "mean_fold_auroc": metrics["mean_fold_auroc"],
                        "pooled_auroc": metrics["pooled_auroc"],
                    }
                )
    _write_csv(cohort_rows, table_dir / "cohort_auroc.csv")

    domain_rows = []
    baseline = artifact["cohort_analyses"]["all_nine_domains"][
        "full_training_and_evaluation_refit"
    ]
    for probe, metrics in baseline.items():
        for fold in metrics["folds"]:
            domain_rows.append(
                {
                    "domain_id": fold["domain_id"],
                    "domain_label": DOMAIN_LABELS[fold["domain_id"]],
                    "probe": probe,
                    "layer": PRIMARY_LAYER,
                    "thinking_mode": PRIMARY_THINKING_MODE,
                    "agent_id": PRIMARY_AGENT,
                    "fold_auroc": fold["auroc"],
                    "n_clean": fold["n_clean"],
                    "n_injected": fold["n_injected"],
                }
            )
    _write_csv(domain_rows, table_dir / "domain_layer40_metrics.csv")

    delta_rows = []
    for domain in artifact["paired_injected_minus_clean_scores"]:
        for pair in domain["pairs"]:
            delta_rows.append(
                {
                    "domain_id": domain["domain_id"],
                    "domain_label": DOMAIN_LABELS[domain["domain_id"]],
                    "delegation_depth": pair["delegation_depth"],
                    "clean_score": pair["clean_score"],
                    "injected_score": pair["injected_score"],
                    "injected_minus_clean": pair["injected_minus_clean"],
                    "domain_mean_delta": domain["mean_injected_minus_clean"],
                    "domain_observed_min": domain["observed_min"],
                    "domain_observed_max": domain["observed_max"],
                }
            )
    _write_csv(delta_rows, table_dir / "paired_score_deltas.csv")
    _write_csv(artifact["design_covariates"], table_dir / "design_covariates.csv")


def _plot_paired_deltas(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row["mean_injected_minus_clean"])
    labels = [DOMAIN_LABELS[row["domain_id"]] for row in ordered]
    y_positions = list(range(len(ordered)))
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for y_position, row in zip(y_positions, ordered):
        axis.hlines(
            y_position,
            row["observed_min"],
            row["observed_max"],
            color="#7A8798",
            linewidth=1.4,
            zorder=1,
        )
        pair_values = [pair["injected_minus_clean"] for pair in row["pairs"]]
        axis.scatter(
            pair_values,
            [y_position] * len(pair_values),
            marker="x",
            color="#D55E00",
            s=28,
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            [row["mean_injected_minus_clean"]],
            [y_position],
            marker="o",
            color="#0072B2",
            edgecolor="white",
            linewidth=0.6,
            s=42,
            zorder=3,
        )
    axis.axvline(0.0, color="#374151", linewidth=0.8, linestyle="--")
    axis.set_yticks(y_positions, labels)
    axis.set_xlabel(f"Layer-{PRIMARY_LAYER} score change (injected − clean)")
    axis.set_title("Worker 1, thinking off: paired construction-score changes")
    axis.text(
        0.0,
        -0.16,
        "Circle = mean; × = each depth; line = observed two-pair range (not a confidence interval).",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"worker1_layer40_paired_score_deltas.{suffix}"
        save_reproducible_figure(
            fig,
            path,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": "Worker 1 paired construction-score changes",
                "Author": "SPEC-GAP",
            },
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _render_markdown(artifact: dict[str, Any]) -> str:
    primary = artifact["cohort_analyses"]["all_nine_domains"][
        "full_training_and_evaluation_refit"
    ][PRIMARY_PROBE]
    no_special = artifact["cohort_analyses"]["remove_kg_and_convex"]
    plain = artifact["cohort_analyses"]["plain_text_six_domains"]
    residual = artifact["train_fold_only_domain_mean_residualization"]["results"][
        PRIMARY_PROBE
    ]
    permutation = artifact["permutation_null"]["mean_fold_auroc_null"]
    lines = [
        "# Cross-domain construction-signal robustness",
        "",
        "This is a sensitivity analysis of the saved Scenario 1 activations. It made **no new model calls** and did not redesign or rerun Scenario 1.",
        "",
        f"The frozen source cohort is classified as **`{artifact['analysis_tier']}`** because its trajectories predate execution-tier tagging; this is not a definitive-run claim.",
        "",
        "## What the headline means",
        "",
        f"Worker 1, thinking off, layer {PRIMARY_LAYER} reaches **{primary['mean_fold_auroc']:.3f} mean held-out-domain AUROC** for the `injection_present` construction label. The pooled AUROC is {primary['pooled_auroc']:.3f}. This detects the presence of injected prompt construction/tokens; it is not compromise detection. All 36 injected runs resisted, so compromise-detection AUROC cannot be estimated from this sample.",
        "",
        "AIHC is the only 0.0 held-out fold; the other eight folds are 1.0. With four observations in each domain fold, those estimates are coarse and fragile.",
        "",
        "## Style and exposure sensitivity",
        "",
        "Six domains use plain text; Macro alone wraps text in `<think>`; KG and Convex use chat special tokens plus explicit tool syntax. Seven domains use natural carrier selection. Policy and Telecom require a clean anchor and were position-adjusted.",
        "",
        "| Cohort | Domains | Fold filtering only | Full training/evaluation re-fit |",
        "|---|---:|---:|---:|",
        f"| All domains | 9 | {primary['mean_fold_auroc']:.3f} | {primary['mean_fold_auroc']:.3f} |",
        f"| Remove KG + Convex (Macro remains) | 7 | {no_special['existing_nine_domain_fits_with_folds_filtered'][PRIMARY_PROBE]['mean_fold_auroc']:.3f} | {no_special['full_training_and_evaluation_refit'][PRIMARY_PROBE]['mean_fold_auroc']:.3f} |",
        f"| Exact plain-text subset | 6 | {plain['existing_nine_domain_fits_with_folds_filtered'][PRIMARY_PROBE]['mean_fold_auroc']:.3f} | {plain['full_training_and_evaluation_refit'][PRIMARY_PROBE]['mean_fold_auroc']:.3f} |",
        "",
        "The fold-filtered column preserves probes trained with all nine domains. The re-fit column removes excluded domains from both training and evaluation; it is the stricter ablation.",
        "",
        "## Requested robustness checks",
        "",
        f"- Train-fold-only domain-mean residualization changes Goldowsky-Dill mean fold AUROC from {residual['before']['mean_fold_auroc']:.3f} to {residual['after']['mean_fold_auroc']:.3f}. Held-out values never fit the transform.",
        f"- The end-to-end balanced within-domain permutation null re-fits every probe. Its add-one p-value for mean fold AUROC is {permutation['add_one_p_value']:.4f} across {artifact['permutation_null']['n_permutations']} deterministic permutations.",
        "- The paired-delta forest plot shows both depth-specific points and their mean. Each domain has only two pairs, so the plotted range is not a confidence interval.",
        "",
        "![Paired score deltas](figures/worker1_layer40_paired_score_deltas.png)",
        "",
        "## Interpretation boundary",
        "",
        "These checks show that the construction signal persists but changes when attack style and domain baselines are altered. Worker 2 and executor estimates are also based on only nine held-out domains and small per-mode sample counts. A new combined natural-text attack, mechanism axis, or arbitrary tool target would require a future Scenario 1 redesign by the research group; it is not part of this cleanup.",
        "",
    ]
    return "\n".join(lines)


def _groups_from_metadata(metadata: list[dict[str, Any]]):
    import numpy as np

    return np.asarray([str(row["match_group_id"]) for row in metadata])


def _index_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["match_group_id"]),
        str(row["delegation_depth"]),
        str(row["treatment"]),
        str(row["trajectory_id"]),
        int(row["hop_index"]),
    )


def _score_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
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


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
