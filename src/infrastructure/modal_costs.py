"""Pure helpers for per-input Modal GPU cost and token-usage records.

These records are estimates for experiment tracking. Modal's billing report is
the source of truth for invoiced workspace spend, credits, and shared container
overhead.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import PurePosixPath
from typing import Any

from src.infrastructure.model_profiles import ACTIVE_PROFILE


# Modal list prices per GPU-second. The "h200" in the record field names below
# predates other GPUs; ``gpu_type`` and ``gpu_rate_usd_per_second`` say which
# GPU and rate a record's estimate actually uses.
GPU_USD_PER_SECOND = {
    "H200": 0.001261,
    "A100-80GB": 0.000694,
}
GPU_TYPE = ACTIVE_PROFILE.gpu
GPU_COUNT = 1
H200_USD_PER_SECOND = GPU_USD_PER_SECOND["H200"]
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_CHECKED_ON = "2026-10-28"
COST_SCHEMA_VERSION = "spec_gap.modal_gpu_cost.v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CostRecordValidationError(ValueError):
    """Raised when a local cost or token-usage record is inconsistent."""


def resolve_modal_input_id(
    input_id: str | None,
    function_call_id: str | None,
) -> str:
    """Choose a safe Modal identifier for a per-input cost artifact.

    Modal may not expose ``current_input_id()`` inside every class method. In
    that case, the function-call ID still gives the invocation a stable,
    collision-resistant name. Invalid values are never written into a path.
    """

    for value in (input_id, function_call_id):
        if isinstance(value, str) and _SAFE_ID.fullmatch(value):
            return value
    raise CostRecordValidationError(
        "Modal did not provide a safe input ID or function-call ID"
    )


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CostRecordValidationError(f"{field} must be a non-negative integer")
    return value


def build_token_usage(
    *,
    input_token_count: int,
    generated_token_count: int,
    thinking_token_count: int,
    final_output_token_count: int,
) -> dict[str, int]:
    """Build a token partition without storing any prompt or response text."""

    input_count = _non_negative_int(input_token_count, "input_token_count")
    generated_count = _non_negative_int(
        generated_token_count, "generated_token_count"
    )
    thinking_count = _non_negative_int(
        thinking_token_count, "thinking_token_count"
    )
    final_count = _non_negative_int(
        final_output_token_count, "final_output_token_count"
    )
    if thinking_count + final_count > generated_count:
        raise CostRecordValidationError(
            "thinking and final-output token counts cannot exceed generated tokens"
        )
    return {
        "input_tokens": input_count,
        "generated_tokens": generated_count,
        "thinking_tokens": thinking_count,
        "final_output_tokens": final_count,
        "special_or_separator_tokens": (
            generated_count - thinking_count - final_count
        ),
        "total_tokens_processed": input_count + generated_count,
    }


def cost_record_path(
    *,
    trajectory_id: str,
    thinking_mode: str,
    step_index: int,
    modal_input_id: str,
) -> str:
    """Return a collision-resistant path inside the shared artifact Volume."""

    for field, value in (
        ("trajectory_id", trajectory_id),
        ("thinking_mode", thinking_mode),
        ("modal_input_id", modal_input_id),
    ):
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
            raise CostRecordValidationError(f"{field} is not a safe identifier")
    _non_negative_int(step_index, "step_index")
    return PurePosixPath(
        "costs",
        trajectory_id,
        thinking_mode,
        f"step_{step_index:03d}",
        f"{modal_input_id}.json",
    ).as_posix()


def build_gpu_cost_record(
    *,
    trajectory_id: str,
    step_index: int,
    agent_id: str,
    thinking_mode: str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    modal_input_id: str,
    modal_task_id: str | None,
    token_usage: dict[str, int],
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a per-input GPU estimate with explicit billing caveats."""

    if not isinstance(elapsed_seconds, (int, float)) or isinstance(
        elapsed_seconds, bool
    ):
        raise CostRecordValidationError("elapsed_seconds must be numeric")
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise CostRecordValidationError(
            "elapsed_seconds must be finite and non-negative"
        )
    if not isinstance(started_at, str) or not started_at:
        raise CostRecordValidationError("started_at must be a non-empty string")
    if not isinstance(finished_at, str) or not finished_at:
        raise CostRecordValidationError("finished_at must be a non-empty string")
    if not isinstance(agent_id, str) or not agent_id:
        raise CostRecordValidationError("agent_id must be a non-empty string")

    normalized_usage = build_token_usage(
        input_token_count=token_usage.get("input_tokens"),
        generated_token_count=token_usage.get("generated_tokens"),
        thinking_token_count=token_usage.get("thinking_tokens"),
        final_output_token_count=token_usage.get("final_output_tokens"),
    )
    generated_tokens = normalized_usage["generated_tokens"]
    rounded_elapsed = round(elapsed, 6)
    rate = GPU_USD_PER_SECOND[GPU_TYPE]
    estimated_cost = rounded_elapsed * rate * GPU_COUNT
    tokens_per_second = (
        generated_tokens / rounded_elapsed if rounded_elapsed > 0 else None
    )
    cost_per_1k = (
        estimated_cost * 1000 / generated_tokens
        if generated_tokens > 0
        else None
    )
    storage_path = cost_record_path(
        trajectory_id=trajectory_id,
        thinking_mode=thinking_mode,
        step_index=step_index,
        modal_input_id=modal_input_id,
    )
    record = {
        "schema_version": COST_SCHEMA_VERSION,
        "cost_kind": "per_input_h200_estimate",
        "billing_status": "estimate_pending_workspace_reconciliation",
        "trajectory_id": trajectory_id,
        "step_index": step_index,
        "agent_id": agent_id,
        "thinking_mode": thinking_mode,
        "modal_input_id": modal_input_id,
        "modal_task_id": modal_task_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": rounded_elapsed,
        "gpu_type": GPU_TYPE,
        "gpu_count": GPU_COUNT,
        "gpu_rate_usd_per_second": rate,
        "estimated_h200_cost_usd": round(estimated_cost, 8),
        "generated_tokens_per_second": (
            round(tokens_per_second, 6)
            if tokens_per_second is not None
            else None
        ),
        "estimated_h200_cost_per_1k_generated_tokens_usd": (
            round(cost_per_1k, 8) if cost_per_1k is not None else None
        ),
        "currency": "USD",
        "price_source": PRICE_SOURCE,
        "price_checked_on": PRICE_CHECKED_ON,
        "token_usage": normalized_usage,
        "runtime_metadata": copy.deepcopy(runtime_metadata or {}),
        "storage_path": storage_path,
        "includes": [
            f"measured Qwen model-turn method time on one {GPU_TYPE}",
        ],
        "excludes": [
            "container startup and model-load time before this input",
            "warm-container idle time",
            "CPU and memory charges",
            "Volume storage and network charges",
            "workspace credits, reservations, and discounts",
        ],
    }
    return validate_gpu_cost_record(record)


def validate_gpu_cost_record(payload: Any) -> dict[str, Any]:
    """Validate a saved estimate without calling Modal's billing service."""

    if not isinstance(payload, dict):
        raise CostRecordValidationError("cost record must be an object")
    record = copy.deepcopy(payload)
    if record.get("schema_version") != COST_SCHEMA_VERSION:
        raise CostRecordValidationError(
            f"schema_version must be {COST_SCHEMA_VERSION!r}"
        )
    if record.get("cost_kind") != "per_input_h200_estimate":
        raise CostRecordValidationError(
            "cost_kind must be 'per_input_h200_estimate'"
        )
    gpu_type = record.get("gpu_type")
    if gpu_type not in GPU_USD_PER_SECOND or record.get("gpu_count") != GPU_COUNT:
        raise CostRecordValidationError(
            f"cost record must describe one GPU of {sorted(GPU_USD_PER_SECOND)}"
        )
    rate = GPU_USD_PER_SECOND[gpu_type]
    if record.get("gpu_rate_usd_per_second") != rate:
        raise CostRecordValidationError(
            f"{gpu_type} rate does not match the pinned estimate"
        )

    usage = record.get("token_usage")
    if not isinstance(usage, dict):
        raise CostRecordValidationError("token_usage must be an object")
    normalized_usage = build_token_usage(
        input_token_count=usage.get("input_tokens"),
        generated_token_count=usage.get("generated_tokens"),
        thinking_token_count=usage.get("thinking_tokens"),
        final_output_token_count=usage.get("final_output_tokens"),
    )
    if usage != normalized_usage:
        raise CostRecordValidationError("token_usage totals are inconsistent")

    elapsed = record.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool):
        raise CostRecordValidationError("elapsed_seconds must be numeric")
    elapsed = float(elapsed)
    expected_cost = round(elapsed * rate * GPU_COUNT, 8)
    if record.get("estimated_h200_cost_usd") != expected_cost:
        raise CostRecordValidationError("estimated GPU cost is inconsistent")
    expected_path = cost_record_path(
        trajectory_id=record.get("trajectory_id"),
        thinking_mode=record.get("thinking_mode"),
        step_index=record.get("step_index"),
        modal_input_id=record.get("modal_input_id"),
    )
    if record.get("storage_path") != expected_path:
        raise CostRecordValidationError("cost storage_path is inconsistent")
    return record
