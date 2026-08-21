"""Build and resume the controlled live Scenario 1 run matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.infrastructure.qwen_modal import ACTIVATION_ARTIFACT_FORMAT
from src.infrastructure.modal_billing import validate_analysis_tier
from src.scenario1.generator import build_record
from src.scenario1.validator import validate_payload


THINKING_MODES = ("off", "on")


def has_current_activation_format(record: dict[str, Any]) -> bool:
    """Return whether every model turn has the required position checkpoints."""

    turns = [
        event
        for event in record.get("trajectory_trace", {}).get("full_events", [])
        if event.get("type") == "agent_turn"
    ]
    if not turns:
        return False
    for turn in turns:
        activation = turn.get("activation_metadata")
        if not isinstance(activation, dict):
            return False
        if activation.get("artifact_format_version") != ACTIVATION_ARTIFACT_FORMAT:
            return False
        checkpoints = activation.get("checkpoint_positions")
        if not isinstance(checkpoints, list):
            return False
        names = {
            checkpoint.get("name")
            for checkpoint in checkpoints
            if isinstance(checkpoint, dict)
        }
        required = {"last_input_token"}
        output = turn.get("output", {})
        if output.get("thinking_content"):
            required.add("last_reasoning_token")
        if output.get("final_content"):
            required.add("last_visible_answer_token")
        if not required.issubset(names):
            return False
        forward_scopes = activation.get("checkpoint_forward_scopes")
        if not isinstance(forward_scopes, dict):
            return False
        expected_scopes = {
            name: (
                "prompt_only" if name == "last_input_token" else "generated_prefix"
            )
            for name in names
        }
        if forward_scopes != expected_scopes:
            return False
    return True


def build_live_batch(
    registries: Iterable[dict[str, Any]],
    *,
    thinking_modes: Iterable[str] = THINKING_MODES,
    output_root: str | Path,
    analysis_tier: str = "exploratory",
) -> list[dict[str, Any]]:
    """Return the deterministic live-run plan for the supplied registries."""

    analysis_tier = validate_analysis_tier(analysis_tier)
    modes = list(thinking_modes)
    if not modes or len(modes) != len(set(modes)):
        raise ValueError("thinking modes must be a non-empty unique list")
    if any(mode not in THINKING_MODES for mode in modes):
        raise ValueError("thinking modes must contain only 'off' and 'on'")

    root = Path(output_root)
    items: list[dict[str, Any]] = []
    for mode in modes:
        for registry in registries:
            for condition in registry["conditions"]:
                structural = build_record(
                    registry,
                    condition["condition_id"],
                    condition["treatment"],
                )
                live_id = f"{structural['trajectory_id']}__thinking_{mode}"
                items.append({
                    "trajectory_id": live_id,
                    "thinking_mode": mode,
                    "analysis_tier": analysis_tier,
                    "condition_id": condition["condition_id"],
                    "treatment": condition["treatment"],
                    "structural_record": structural,
                    "output_path": (
                        root
                        / "live"
                        / analysis_tier
                        / mode
                        / f"{live_id}.json"
                    ),
                })
    return items


def load_completed_batch_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Load a valid matching live result, or return None when it is pending."""

    path = Path(item["output_path"])
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume from invalid JSON at {path}: {error}") from error

    errors = validate_payload(record)
    if errors:
        raise ValueError(
            f"cannot resume from invalid trajectory at {path}: " + "; ".join(errors)
        )
    expected = {
        "trajectory_id": item["trajectory_id"],
        "condition_id": item["condition_id"],
        "treatment": item["treatment"],
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"cannot resume from {path}: {field} does not match the batch plan"
            )
    if record.get("generation_mode") != "real" or record.get("model_called") is not True:
        raise ValueError(f"cannot resume from {path}: record is not a real model run")
    if record.get("model", {}).get("thinking_mode") != item["thinking_mode"]:
        raise ValueError(f"cannot resume from {path}: thinking mode does not match")
    turn_tiers = {
        event.get("model_execution_metadata", {}).get("analysis_tier")
        for event in record["trajectory_trace"]["full_events"]
        if event.get("type") == "agent_turn"
    }
    if turn_tiers != {item["analysis_tier"]}:
        raise ValueError(
            f"cannot resume from {path}: analysis tier does not match"
        )
    if not has_current_activation_format(record):
        return None
    return record


def trajectory_turn_cost(record: dict[str, Any]) -> float:
    """Sum the per-turn H200 estimates stored in one live trajectory."""

    return sum(
        float(event.get("cost_metadata", {}).get("estimated_h200_cost_usd", 0.0))
        for event in record["trajectory_trace"]["full_events"]
        if event.get("type") == "agent_turn"
    )
