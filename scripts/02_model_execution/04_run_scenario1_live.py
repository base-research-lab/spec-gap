#!/usr/bin/env python3
"""Step 4: validate or run one complete live Scenario 1 trajectory."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.modal_billing import (  # noqa: E402
    scenario1_billing_tags,
    validate_analysis_tier,
)
from src.infrastructure.modal_qwen_runner import Qwen3Runner, app  # noqa: E402
from src.scenario1.generator import (  # noqa: E402
    ARTIFACT_ROOT,
    build_record,
    build_request_plan,
    load_registry,
)
from src.scenario1.orchestrator import (  # noqa: E402
    run_live_trajectory,
    write_live_trajectory,
    write_model_turn_result,
)


@app.local_entrypoint()
def run_scenario1_trajectory(
    registry_path: str,
    condition_id: str = "2-hop",
    treatment: str = "clean",
    thinking_mode: str = "off",
    action: str = "validate",
    output_root: str = str(Path(ARTIFACT_ROOT) / "trajectories"),
    confirm_paid_run: str = "",
    analysis_tier: str = "exploratory",
) -> None:
    """Build the plan without compute, or run one guarded paid trajectory."""

    if condition_id not in {"2-hop", "3-hop"}:
        raise ValueError("condition_id must be '2-hop' or '3-hop'")
    if treatment not in {"clean", "injected"}:
        raise ValueError("treatment must be 'clean' or 'injected'")
    if thinking_mode not in {"on", "off"}:
        raise ValueError("thinking_mode must be 'on' or 'off'")
    analysis_tier = validate_analysis_tier(analysis_tier)

    registry = load_registry(registry_path)
    structural = build_record(registry, condition_id, treatment)
    request_plan = build_request_plan(
        [structural],
        [thinking_mode],
        analysis_tier=analysis_tier,
    )

    if action == "validate":
        print(json.dumps({
            "action": "validate",
            "model_called": False,
            "gpu_started": False,
            "trajectory_id": structural["trajectory_id"],
            "condition_id": condition_id,
            "treatment": treatment,
            "thinking_mode": thinking_mode,
            "analysis_tier": analysis_tier,
            "agent_order": [request["agent_id"] for request in request_plan],
            "model_turn_count": len(request_plan),
            "activation_layers": request_plan[0]["activation_layers"],
        }, indent=2))
        print("Validation passed. No remote function or GPU was started.")
        return
    if action != "run":
        raise ValueError("action must be validate or run")
    if confirm_paid_run != "RUN_H200_TRAJECTORY":
        raise ValueError(
            "paid trajectory execution requires "
            "--confirm-paid-run RUN_H200_TRAJECTORY"
        )

    billing_tags = scenario1_billing_tags(
        domain_ids=[structural["domain_id"]],
        generation_protocol_ids=[structural["generation_protocol_id"]],
        run_kind="single_trajectory",
        analysis_tier=analysis_tier,
    )
    app.set_tags(billing_tags)
    print(json.dumps({
        "modal_app_id": app.app_id,
        "billing_tags": billing_tags,
    }, indent=2))

    runner = Qwen3Runner()
    live = run_live_trajectory(
        structural,
        thinking_mode=thinking_mode,
        analysis_tier=analysis_tier,
        generate_turn=runner.generate_agent_turn.remote,
        on_turn_complete=lambda _request, result: write_model_turn_result(
            result, output_root
        ),
    )
    path = write_live_trajectory(live, output_root)
    turn_costs = [
        event.get("cost_metadata", {}).get("estimated_h200_cost_usd", 0.0)
        for event in live["trajectory_trace"]["full_events"]
        if event.get("type") == "agent_turn"
        and isinstance(event.get("cost_metadata"), dict)
    ]
    print(json.dumps({
        "trajectory_id": live["trajectory_id"],
        "output_path": str(path),
        "condition_id": live["condition_id"],
        "treatment": live["treatment"],
        "thinking_mode": live["model"]["thinking_mode"],
        "model_turn_count": len(turn_costs),
        "outcome_class": live["evaluation_labels"]["outcome_class"],
        "unsafe_action_executed": live["evaluation_labels"]["action_channel"][
            "unsafe_action_executed"
        ],
        "estimated_model_turn_h200_cost_usd": round(sum(turn_costs), 8),
        "billing_note": (
            "Per-turn estimate excludes model startup, CPU, memory, storage, "
            "credits, and discounts. Modal billing is the final source."
        ),
    }, indent=2))


__all__ = ["app", "run_scenario1_trajectory"]
