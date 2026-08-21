#!/usr/bin/env python3
"""Reconcile local Scenario 1 estimates with Modal's billing source of truth."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.modal_billing import (  # noqa: E402
    ANALYSIS_TIERS,
    BILLING_RECONCILIATION_SCHEMA_VERSION,
    aggregate_billing_rows,
    as_decimal,
    decimal_text,
    validate_analysis_tier,
)


APP_ID_RE = re.compile(r"\bap-[A-Za-z0-9]+\b")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_modal_billing(
    modal_module: Any,
    *,
    start: datetime,
    end: datetime,
    billing_cycle: str,
) -> tuple[list[Any], Any]:
    """Load Modal billing data with actionable authentication guidance."""

    try:
        workspace = modal_module.Workspace.from_context()
        rows = workspace.billing.report(
            start=start,
            end=end,
            resolution="h",
            tag_names=["*"],
        )
        summary = workspace.billing.summary(billing_cycle)
    except modal_module.exception.AuthError as exc:
        raise SystemExit(
            "Modal authentication failed. Run `modal setup`, then confirm "
            "the active profile with `modal profile current`."
        ) from exc
    return rows, summary


def _local_estimates(
    checkpoint_root: Path,
    *,
    analysis_tier: str,
) -> dict[str, Any]:
    by_input: dict[str, dict[str, Any]] = {}
    reference_count = 0
    excluded_unclassified_count = 0
    excluded_other_tier_count = 0
    for path in sorted(checkpoint_root.rglob("step_*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cost = payload.get("cost_metadata")
        if not isinstance(cost, dict):
            continue
        runtime = cost.get("runtime_metadata")
        saved_tier = (
            runtime.get("analysis_tier")
            if isinstance(runtime, dict) else None
        )
        if saved_tier is None:
            excluded_unclassified_count += 1
            continue
        if saved_tier != analysis_tier:
            excluded_other_tier_count += 1
            continue
        modal_input_id = cost.get("modal_input_id")
        estimate = cost.get("estimated_h200_cost_usd")
        if not isinstance(modal_input_id, str) or estimate is None:
            continue
        reference_count += 1
        existing = by_input.get(modal_input_id)
        if existing is not None and existing != cost:
            raise ValueError(
                f"conflicting local cost records for {modal_input_id}"
            )
        by_input[modal_input_id] = cost
    total = sum(
        (
            as_decimal(item["estimated_h200_cost_usd"], "local estimate")
            for item in by_input.values()
        ),
        Decimal(0),
    )
    return {
        "checkpoint_cost_reference_count": reference_count,
        "unique_modal_input_count": len(by_input),
        "duplicate_reference_count": reference_count - len(by_input),
        "analysis_tier": analysis_tier,
        "excluded_unclassified_checkpoint_count": (
            excluded_unclassified_count
        ),
        "excluded_other_tier_checkpoint_count": excluded_other_tier_count,
        "estimated_model_turn_h200_cost_usd": decimal_text(total),
    }


def _local_app_references(results_root: Path) -> dict[str, list[str]]:
    references: dict[str, set[str]] = {}
    for path in sorted(results_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {
            ".json", ".md", ".csv", ".txt"
        }:
            continue
        if "modal_billing_reconciliation" in path.name:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for app_id in APP_ID_RE.findall(text):
            references.setdefault(app_id, set()).add(relative)
    return {
        app_id: sorted(paths) for app_id, paths in references.items()
    }


def _normalized_modal_rows(
    rows: list[Any],
    *,
    project_tag: str,
    analysis_tier: str,
) -> list[dict[str, Any]]:
    project_rows = []
    tiers_by_app: dict[str, set[str | None]] = {}
    for row in rows:
        if row.tags.get("project") != project_tag:
            continue
        tags = dict(row.tags)
        tiers_by_app.setdefault(row.object_id, set()).add(
            tags.get("analysis_tier")
        )
        project_rows.append({
            "object_id": row.object_id,
            "description": row.description,
            "environment": row.environment_name,
            "interval_start": row.interval_start.isoformat(),
            "cost": decimal_text(row.cost),
            "cost_by_resource": {
                name: decimal_text(value)
                for name, value in row.cost_by_resource.items()
            },
            "tags": tags,
        })
    changed = {
        app_id: tiers
        for app_id, tiers in tiers_by_app.items()
        if len(tiers) > 1
    }
    if changed:
        raise ValueError(
            "analysis_tier changed within Modal Apps: "
            + ", ".join(sorted(changed))
        )
    return [
        row for row in project_rows
        if row["tags"].get("analysis_tier") == analysis_tier
    ]


def _workspace_summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "cycle_start": summary.start.isoformat(),
        "cycle_end": summary.end.isoformat(),
        "metered_cost_usd": decimal_text(summary.metered_cost),
        "billed_cost_usd": decimal_text(summary.billed_cost),
        "adjustments_usd": {
            name: decimal_text(value)
            for name, value in sorted(summary.adjustments.items())
        },
        "metered_cost_breakdown_usd": {
            name: decimal_text(value)
            for name, value in sorted(
                summary.metered_cost_breakdown.items()
            )
        },
        "scope_note": (
            "Workspace-wide billing-cycle summary. Credits and adjustments "
            "are not allocated to individual Apps by Modal's granular report."
        ),
    }


def _write_csv(path: Path, apps: list[dict[str, Any]]) -> None:
    resources = sorted({
        resource
        for app in apps
        for resource in app["cost_by_resource_usd"]
    })
    fields = [
        "modal_app_id", "description", "environment",
        "first_interval_start", "last_interval_start",
        "hourly_row_count", "metered_cost_usd",
        *[f"resource_{name}_cost_usd" for name in resources],
        "tags_json", "local_references_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for app in apps:
            writer.writerow({
                **{
                    field: app.get(field, "")
                    for field in fields
                    if not field.startswith("resource_")
                },
                **{
                    f"resource_{name}_cost_usd": app[
                        "cost_by_resource_usd"
                    ].get(name, "0")
                    for name in resources
                },
                "tags_json": json.dumps(app["tags"], sort_keys=True),
                "local_references_json": json.dumps(
                    app["local_references"]
                ),
            })


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    project = payload["project_metered_cost"]
    local = payload["local_estimate"]
    comparison = payload["comparison"]
    workspace = payload["workspace_billing_cycle"]
    percent = comparison["percent_above_estimate"]
    difference_line = (
        "- Metered minus estimate: "
        f"`${comparison['metered_minus_estimate_usd']}`"
    )
    if percent is None:
        difference_line += " (percentage unavailable: no selected local estimate)"
    else:
        difference_line += f" (`{percent}%`)"
    lines = [
        f"# Scenario 1 Modal billing reconciliation — {payload['report_date']}",
        "",
        f"Analysis tier: `{payload['analysis_tier']}`",
        "",
        "## Authoritative totals",
        "",
        (
            "- Modal-metered SPEC-GAP resource cost: "
            f"`${project['metered_cost_usd']}`"
        ),
        (
            "- Local per-turn H200 estimate: "
            f"`${local['estimated_model_turn_h200_cost_usd']}`"
        ),
        difference_line,
        (
            "- Workspace billed cost after July adjustments: "
            f"`${workspace['billed_cost_usd']}`"
        ),
        (
            "- Complete workspace metered cost for the July cycle: "
            f"`${workspace['metered_cost_usd']}`"
        ),
        "",
        (
            "The project total comes directly from Modal's billing API and "
            "includes H200, CPU, and memory only for Apps carrying both the "
            "project tag and the selected analysis-tier tag."
        ),
        (
            "The billed-cost value is workspace-wide; Modal does not allocate "
            "credits to individual Apps in its granular report. The complete "
            "workspace metered total includes usage outside this SPEC-GAP report."
        ),
        "",
        "## Resource breakdown",
        "",
        "| Resource | Modal-metered cost |",
        "|---|---:|",
    ]
    for name, value in project["cost_by_resource_usd"].items():
        lines.append(f"| {name} | `${value}` |")
    lines.extend([
        "",
        "## Coverage",
        "",
        f"- Modal Apps: `{project['modal_app_count']}`",
        f"- Hourly billing rows: `{project['hourly_billing_row_count']}`",
        (
            "- Unique saved model inputs: "
            f"`{local['unique_modal_input_count']}`"
        ),
        (
            "- Excluded unclassified checkpoints: "
            f"`{local['excluded_unclassified_checkpoint_count']}`"
        ),
        (
            "- Excluded other-tier checkpoints: "
            f"`{local['excluded_other_tier_checkpoint_count']}`"
        ),
        (
            "- Apps with a local report reference: "
            f"`{payload['mapping_coverage']['referenced_app_count']}`"
        ),
        (
            "- Apps needing a retrospective run label: "
            f"`{payload['mapping_coverage']['unreferenced_app_count']}`"
        ),
        "",
        "## Interpretation",
        "",
        (
            "Use the Modal-metered total for infrastructure-cost reporting. "
            "Use the local estimate only for per-turn and per-trajectory "
            "diagnostics. For future runs, Modal App tags provide domain and "
            "generation-protocol attribution."
        ),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--billing-cycle", required=True)
    parser.add_argument("--project-tag", default="spec-gap")
    parser.add_argument(
        "--analysis-tier",
        required=True,
        choices=sorted(ANALYSIS_TIERS),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/scenario1/trajectories",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    try:
        import modal
    except ImportError as exc:
        raise SystemExit(
            "Install the optional Modal dependencies: uv sync --extra modal"
        ) from exc

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end) if args.end else datetime.now(timezone.utc)
    analysis_tier = validate_analysis_tier(args.analysis_tier)
    report_rows, workspace_summary = _load_modal_billing(
        modal,
        start=start,
        end=end,
        billing_cycle=args.billing_cycle,
    )
    normalized = _normalized_modal_rows(
        report_rows,
        project_tag=args.project_tag,
        analysis_tier=analysis_tier,
    )
    if not normalized:
        raise ValueError(
            "no Modal billing rows matched project="
            f"{args.project_tag!r} and analysis_tier={analysis_tier!r}"
        )
    apps, project_summary = aggregate_billing_rows(normalized)
    local_estimate = _local_estimates(
        args.checkpoint_root,
        analysis_tier=analysis_tier,
    )
    references = _local_app_references(args.results_root)
    for app in apps:
        app["local_references"] = references.get(app["modal_app_id"], [])

    metered = as_decimal(project_summary["metered_cost_usd"], "metered cost")
    estimated = as_decimal(
        local_estimate["estimated_model_turn_h200_cost_usd"],
        "local estimate",
    )
    difference = metered - estimated
    percent = difference * Decimal(100) / estimated if estimated else None
    payload = {
        "schema_version": BILLING_RECONCILIATION_SCHEMA_VERSION,
        "analysis_tier": analysis_tier,
        "report_date": datetime.now(timezone.utc).date().isoformat(),
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Modal",
            "api": "Workspace.billing.report and Workspace.billing.summary",
            "source_of_truth": True,
            "report_scope": "full hourly intervals only",
        },
        "query": {
            "start": start.isoformat(),
            "end_requested": end.isoformat(),
            "resolution": "h",
            "required_tags": {
                "project": args.project_tag,
                "analysis_tier": analysis_tier,
            },
        },
        "project_metered_cost": project_summary,
        "local_estimate": local_estimate,
        "comparison": {
            "metered_minus_estimate_usd": decimal_text(difference),
            "percent_above_estimate": (
                decimal_text(percent.quantize(Decimal("0.01")))
                if percent is not None else None
            ),
        },
        "workspace_billing_cycle": _workspace_summary_payload(
            workspace_summary
        ),
        "mapping_coverage": {
            "referenced_app_count": sum(bool(x["local_references"]) for x in apps),
            "unreferenced_app_count": sum(
                not x["local_references"] for x in apps
            ),
            "note": (
                "Apps missing the selected analysis_tier tag are excluded so "
                "exploratory and definitive costs cannot be pooled."
            ),
        },
        "apps": apps,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_csv(args.output_csv, apps)
    _write_markdown(args.output_md, payload)
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "output_md": str(args.output_md),
        "modal_metered_spec_gap_cost_usd": project_summary[
            "metered_cost_usd"
        ],
        "workspace_billed_cost_usd": payload[
            "workspace_billing_cycle"
        ]["billed_cost_usd"],
    }, indent=2))


if __name__ == "__main__":
    main()
