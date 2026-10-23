"""Behavioral-only Scenario 1 runner for hosted black-box models (GLM-5.3 etc.).

Reuses the white-box pipeline's request construction, controlled endpoint,
safe simulated executor, and outcome labelling, so behavioral outcomes are
directly comparable with the Qwen3-32B cohort. What it cannot reproduce:
rendered prompts, token ids, token alignment, and activations. Records are
therefore written under a separate ``behavioral_endpoint`` tree, marked
``white_box_available: false``, and must never enter probe analysis.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.infrastructure.modal_billing import validate_analysis_tier
from src.infrastructure.openai_endpoint import BEHAVIORAL_RESULT_FORMAT
from src.scenario1.generator import build_generation_request
from src.scenario1.orchestrator import (
    _agent_events,
    _append_action_events,
    _behavioral_label,
    _finalize_labels,
    resolve_expected_endpoint,
)
from src.scenario1.safe_executor import execute_simulated_requests


GenerateTurn = Callable[[dict[str, Any]], dict[str, Any]]
TurnComplete = Callable[[dict[str, Any], dict[str, Any]], None]
RECORD_FORMAT = "spec_gap.behavioral_endpoint_trajectory.v1"
_MATCH_FIELDS = (
    "trajectory_id", "step_index", "agent_id", "agent_role", "hop_index",
    "thinking_mode", "analysis_tier", "raw_poison_exposed", "injection_text",
)


def _check_result(request: dict[str, Any], result: dict[str, Any]) -> None:
    if result.get("result_format") != BEHAVIORAL_RESULT_FORMAT:
        raise ValueError("behavioral turn result has an unexpected format")
    for name in _MATCH_FIELDS:
        if result.get(name) != request[name]:
            raise ValueError(f"behavioral result {name} does not match its request")
    if result.get("input_messages") != request["messages"]:
        raise ValueError("behavioral result input_messages do not match request")


def _apply_result(
    event: dict[str, Any],
    result: dict[str, Any],
    *,
    expected_endpoint: str,
) -> None:
    event["output"] = {
        "message": result["final_content"],
        "raw_generated_text": result["raw_content"],
        "generated_token_ids": None,
        "thinking_content": result["thinking_content"],
        "thinking_complete": None,
        "final_content": result["final_content"],
        "parsed_message": {
            "role": "assistant",
            "content": result["final_content"],
            "tool_calls": copy.deepcopy(result["tool_call_requests"]),
        },
        "tool_call_requests": copy.deepcopy(result["tool_call_requests"]),
        "tool_call_parse_errors": copy.deepcopy(result["tool_call_parse_errors"]),
        "finish_reason": result["finish_reason"],
        "truncated": result["truncated"],
    }
    event["exact_model_input"] = {
        "messages": copy.deepcopy(result["input_messages"]),
        "rendered_prompt": None,
        "input_token_ids": None,
        "rendered_prompt_hash": None,
        "request_body_sha256": result["request_body_sha256"],
    }
    event["token_alignment"] = None
    event["activation_metadata"] = None
    event["cost_metadata"] = {
        "billing_basis": "per_token_shared_endpoint",
        "usage": copy.deepcopy(result["usage"]),
        "estimated_cost_usd": result["cost_usd"],
        "latency_s": result["latency_s"],
    }
    event["model_execution_metadata"] = {
        "backend": "openai_compatible_endpoint",
        "model_name": result["model_name"],
        "requested_model": result["requested_model"],
        "endpoint_base_url": result["endpoint_base_url"],
        "response_id": result["response_id"],
        "system_fingerprint": result["system_fingerprint"],
        "thinking_mode": result["thinking_mode"],
        "thinking_mode_violation": result["thinking_mode_violation"],
        "analysis_tier": result["analysis_tier"],
        "generation_settings": copy.deepcopy(result["generation_settings"]),
        "request_body_settings": copy.deepcopy(result["request_body_settings"]),
        "raw_poison_exposed": result["raw_poison_exposed"],
    }
    event["generation_mode"] = "real"
    event["model_called"] = True
    event["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
    event["behavioral_compromise_label"] = _behavioral_label(
        result,
        expected_endpoint=expected_endpoint,
    )
    event["reasoning_compromise_label"] = {
        "label": None,
        "annotation_status": "human_review_required",
        "note": "Black-box endpoint: no latent evidence available.",
    }


def run_behavioral_trajectory(
    structural_record: dict[str, Any],
    *,
    thinking_mode: str,
    generate_turn: GenerateTurn,
    model_label: str,
    on_turn_complete: TurnComplete | None = None,
    analysis_tier: str = "exploratory",
) -> dict[str, Any]:
    """Run every agent in order against a hosted model and label the outcome."""

    if thinking_mode not in {"on", "off"}:
        raise ValueError("thinking_mode must be 'on' or 'off'")
    analysis_tier = validate_analysis_tier(analysis_tier)
    if structural_record.get("generation_mode") != "dry_run":
        raise ValueError("orchestration requires a structural dry-run record")

    record = copy.deepcopy(structural_record)
    construction_id = record["trajectory_id"]
    record["record_format"] = RECORD_FORMAT
    record["white_box_available"] = False
    record["construction_trajectory_id"] = construction_id
    record["trajectory_id"] = f"{construction_id}__thinking_{thinking_mode}"
    record["generation_mode"] = "real"
    record["model_called"] = True
    record["model"]["thinking_mode"] = thinking_mode
    record["model"]["behavioral_model_label"] = model_label
    record["provenance"]["orchestrator"] = "src/scenario1/behavioral_orchestrator.py"
    for event in record["trajectory_trace"]["full_events"]:
        event["trajectory_id"] = record["trajectory_id"]

    expected_endpoint = resolve_expected_endpoint(record)
    upstream: str | None = None
    executor_event: dict[str, Any] | None = None
    for event in _agent_events(record):
        # Built (and validated) by the same code path as the Qwen run, so the
        # messages, tools and injection placement are byte-identical.
        request = build_generation_request(
            record,
            event,
            thinking_mode=thinking_mode,
            upstream_message=upstream,
            analysis_tier=analysis_tier,
        )
        request["extract_activations"] = False
        result = generate_turn(request)
        _check_result(request, result)
        if on_turn_complete is not None:
            on_turn_complete(request, copy.deepcopy(result))
        _apply_result(event, result, expected_endpoint=expected_endpoint)
        upstream = result["downstream_message"]
        if event["agent_id"] == "executor_1":
            executor_event = event

    if executor_event is None:
        raise ValueError("trajectory does not contain executor_1")
    actions = execute_simulated_requests(
        executor_event["output"]["tool_call_requests"],
        expected_destination=expected_endpoint,
        allowed_document_ids={
            document["doc_id"] for document in record["document_set"]["documents"]
        },
    )
    executor_event["output"]["actions"] = actions
    _append_action_events(record, executor_event, actions)
    _finalize_labels(record)
    turns = _agent_events(record)
    record["behavioral_run_summary"] = {
        "outcome_class": record["evaluation_labels"]["outcome_class"],
        "thinking_mode_violations": sum(
            bool(t["model_execution_metadata"]["thinking_mode_violation"])
            for t in turns
        ),
        "estimated_cost_usd": sum(
            t["cost_metadata"]["estimated_cost_usd"] for t in turns
        ),
    }
    return record


def _checkpoint_path(root: Path, request: dict[str, Any]) -> Path:
    return (
        root / "checkpoints" / request["analysis_tier"]
        / request["trajectory_id"] / request["thinking_mode"]
        / f"step_{request['step_index']:03d}.json"
    )


def write_turn(result: dict[str, Any], root: str | Path, request: dict[str, Any]) -> Path:
    path = _checkpoint_path(Path(root), request)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def load_turn(request: dict[str, Any], root: str | Path) -> dict[str, Any] | None:
    path = _checkpoint_path(Path(root), request)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    _check_result(request, result)
    return result


def trajectory_output_path(root: str | Path, tier: str, mode: str, live_id: str) -> Path:
    return Path(root) / "live" / tier / mode / f"{live_id}.json"


def write_trajectory(record: dict[str, Any], root: str | Path, tier: str) -> Path:
    path = trajectory_output_path(
        root, tier, record["model"]["thinking_mode"], record["trajectory_id"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
