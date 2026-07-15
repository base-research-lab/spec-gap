#!/usr/bin/env python3
"""Repair last-input activations without regenerating Scenario 1 outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.modal_qwen_runner import Qwen3Runner, app  # noqa: E402
from src.scenario1.activation_repair import (  # noqa: E402
    apply_activation_repair_result,
    build_activation_repair_plan,
    reconcile_activation_repair_metadata,
    select_activation_repairs,
    summarize_activation_repair_plan,
    validate_local_activation_repair_sources,
    verify_activation_repair_smoke_pair,
)


DEFAULT_TRAJECTORY_ROOT = (
    PROJECT_ROOT / "experiments/scenario1/trajectories/live"
)
DEFAULT_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "experiments/scenario1/trajectories/checkpoints"
)
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "results/scenario1/activation_repair"
REPAIR_RUN_REPORT_SCHEMA = "spec_gap.activation_repair_run.v1"
CONFIRMATIONS = {
    "smoke_pair": "RUN_H200_ACTIVATION_REPAIR_SMOKE",
    "all": "RUN_H200_ACTIVATION_REPAIR_ALL",
}


@app.local_entrypoint()
def repair_prompt_activations(
    action: str = "validate",
    scope: str = "smoke_pair",
    trajectory_root: str = str(DEFAULT_TRAJECTORY_ROOT),
    checkpoint_root: str = str(DEFAULT_CHECKPOINT_ROOT),
    artifact_root: str = str(PROJECT_ROOT),
    report_root: str = str(DEFAULT_REPORT_ROOT),
    max_repairs: int = 0,
    confirm_paid_run: str = "",
) -> None:
    """Validate by default; a paid repair requires a scope-specific guard."""

    if action not in {"validate", "reconcile", "run"}:
        raise ValueError("action must be validate, reconcile, or run")
    if action == "reconcile" and (scope != "all" or max_repairs != 0):
        raise ValueError(
            "metadata reconciliation requires --scope all and no --max-repairs"
        )

    plan = build_activation_repair_plan(trajectory_root, checkpoint_root)
    selected = select_activation_repairs(
        plan,
        scope=scope,
        max_repairs=max_repairs,
    )
    preview = summarize_activation_repair_plan(plan, selected, scope=scope)
    metadata_sync_items = [
        item for item in plan if item["status"] == "metadata_sync_required"
    ]
    source_validation_items = plan if scope == "all" else selected
    if action == "reconcile" and metadata_sync_items:
        source_validation_items = metadata_sync_items
    source_preflight = validate_local_activation_repair_sources(
        source_validation_items,
        artifact_root=artifact_root,
    )
    preview.update({
        "action": action,
        "model_called": False,
        "gpu_started": False,
        "artifact_root": str(Path(artifact_root).resolve()),
        "local_source_preflight": source_preflight,
    })
    print(json.dumps(preview, indent=2), flush=True)

    if action == "reconcile":
        reconciled = [
            reconcile_activation_repair_metadata(
                item,
                artifact_root=artifact_root,
            )
            for item in metadata_sync_items
        ]
        refreshed_plan = build_activation_repair_plan(
            trajectory_root,
            checkpoint_root,
        )
        remaining = [
            item
            for item in refreshed_plan
            if item["status"] == "metadata_sync_required"
        ]
        if remaining:
            raise RuntimeError("activation metadata reconciliation is incomplete")
        print(json.dumps({
            "status": "completed",
            "action": "reconcile",
            "reconciled_turns": len(reconciled),
            "repairs": reconciled,
            "model_called": False,
            "gpu_started": False,
        }, indent=2), flush=True)
        return
    if metadata_sync_items:
        raise ValueError(
            "a prior repair was only partially applied; run --action reconcile "
            "--scope all before starting new GPU work"
        )
    pre_run_smoke_verification = None
    if scope == "all" or (scope == "smoke_pair" and not selected):
        pre_run_smoke_verification = verify_activation_repair_smoke_pair(
            plan,
            artifact_root=artifact_root,
        )
        print(json.dumps(pre_run_smoke_verification, indent=2), flush=True)
        if pre_run_smoke_verification["status"] != "passed":
            raise RuntimeError(
                "the repaired planner pair is not exactly identical; do not run "
                "the full repair"
            )
    if action == "validate":
        print(
            "Repair validation passed. No generation, remote method, or GPU was started.",
            flush=True,
        )
        return
    expected_confirmation = CONFIRMATIONS.get(scope)
    if confirm_paid_run != expected_confirmation:
        raise ValueError(
            "paid repair requires --confirm-paid-run "
            f"{expected_confirmation} for scope={scope}"
        )
    if not selected:
        print("Nothing to repair in the selected scope.", flush=True)
        return

    runner = Qwen3Runner()
    summaries = []
    for index, item in enumerate(selected, start=1):
        identifier = f"{item['trajectory_id']}:step_{item['step_index']:03d}"
        print(f"[{index}/{len(selected)}] repairing {identifier}", flush=True)
        remote_result = runner.repair_prompt_activation.remote(item["request"])
        summary = apply_activation_repair_result(
            item,
            remote_result,
            artifact_root=artifact_root,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)

    total_cost = sum(item["estimated_h200_cost_usd"] for item in summaries)
    verification = None
    if scope == "smoke_pair":
        repaired_plan = build_activation_repair_plan(
            trajectory_root,
            checkpoint_root,
        )
        verification = verify_activation_repair_smoke_pair(
            repaired_plan,
            artifact_root=artifact_root,
        )
        print(json.dumps(verification, indent=2), flush=True)
        if verification["status"] != "passed":
            failure_report = {
                "schema_version": REPAIR_RUN_REPORT_SCHEMA,
                "status": "failed_smoke_verification",
                "repair_method": preview["repair_method"],
                "scope": scope,
                "repair_summaries": summaries,
                "smoke_verification": verification,
                "model_outputs_regenerated": False,
            }
            report_path = _write_repair_report(
                failure_report,
                report_root=report_root,
                scope=scope,
            )
            print(f"Wrote failed repair report to {report_path}", flush=True)
            raise RuntimeError(
                "the repaired planner pair is not exactly identical; do not run "
                "the full repair"
            )
    final = {
        "schema_version": REPAIR_RUN_REPORT_SCHEMA,
        "status": "completed",
        "repair_method": preview["repair_method"],
        "scope": scope,
        "repaired_turns": len(summaries),
        "estimated_h200_method_cost_usd": round(total_cost, 8),
        "model_outputs_regenerated": False,
        "smoke_verification": verification,
        "pre_run_smoke_gate": pre_run_smoke_verification,
        "remaining_pending_repairs": preview["status_counts"]["pending"]
        - len(summaries),
        "billing_note": (
            "Per-input estimates exclude model-load, idle, CPU, memory, storage, "
            "credits, and discounts. Modal billing is the final source."
        ),
    }
    report_path = _write_repair_report(
        {**final, "repair_summaries": summaries},
        report_root=report_root,
        scope=scope,
    )
    final["report_path"] = report_path.as_posix()
    print(json.dumps(final, indent=2), flush=True)


def _write_repair_report(
    payload: dict,
    *,
    report_root: str,
    scope: str,
) -> Path:
    root = Path(report_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scope}_report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["app", "repair_prompt_activations"]
