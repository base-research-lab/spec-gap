"""Pure-Python validation helpers for the Qwen3-32B Modal runner.

This module does not import Modal, Torch, or Transformers. Scenario generators
can validate a request locally before any remote function or GPU is started.
"""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath
from typing import Any


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_LAYER_COUNT = 64
THINKING_MODES = {"on": True, "off": False}
ALLOWED_AGENT_ROLES = {"planner", "worker", "worker2", "executor"}
RAW_POISON_AGENT_ID = "worker_1"

# The thinking-mode ablation changes only enable_thinking. Qwen's recommended
# mode-specific settings can be evaluated separately as a sensitivity check.
CONTROLLED_GENERATION_SETTINGS = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 2048,
    "seed": 0,
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RequestValidationError(ValueError):
    """Raised before a malformed request can start remote model execution."""


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_safe_id(value: str, field: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise RequestValidationError(
            f"{field} may contain only letters, digits, '.', '_' and '-'"
        )


def _validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise RequestValidationError("messages must be a non-empty list")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestValidationError(f"messages[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise RequestValidationError(f"messages[{index}].role must be a string")
        if not isinstance(content, str):
            raise RequestValidationError(f"messages[{index}].content must be a string")
        normalized.append({"role": role.strip(), "content": content})
    return normalized


def _validate_settings(settings: Any) -> dict[str, Any]:
    if settings is None:
        return copy.deepcopy(CONTROLLED_GENERATION_SETTINGS)
    if not isinstance(settings, dict):
        raise RequestValidationError("generation_settings must be an object")

    unknown = sorted(set(settings) - set(CONTROLLED_GENERATION_SETTINGS))
    if unknown:
        raise RequestValidationError(f"unknown generation settings: {unknown}")

    normalized = copy.deepcopy(CONTROLLED_GENERATION_SETTINGS)
    normalized.update(settings)

    if normalized["do_sample"] is not True:
        raise RequestValidationError(
            "the controlled Qwen comparison requires do_sample=true"
        )
    for field in ("temperature", "top_p", "top_k", "min_p"):
        if normalized[field] != CONTROLLED_GENERATION_SETTINGS[field]:
            raise RequestValidationError(
                f"the controlled comparison requires {field}="
                f"{CONTROLLED_GENERATION_SETTINGS[field]!r}; run mode-specific "
                "settings as a separate sensitivity analysis"
            )
    for field in ("temperature", "top_p"):
        value = normalized[field]
        if not isinstance(value, (int, float)) or not 0 < float(value) <= 1:
            raise RequestValidationError(f"{field} must be greater than 0 and at most 1")
    min_p = normalized["min_p"]
    if not isinstance(min_p, (int, float)) or not 0 <= float(min_p) <= 1:
        raise RequestValidationError("min_p must be between 0 and 1")
    top_k = normalized["top_k"]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise RequestValidationError("top_k must be a positive integer")
    max_new_tokens = normalized["max_new_tokens"]
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or not 1 <= max_new_tokens <= 8192
    ):
        raise RequestValidationError("max_new_tokens must be between 1 and 8192")
    seed = normalized["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RequestValidationError("seed must be a non-negative integer")
    return normalized


def _validate_layers(layers: Any) -> list[int]:
    if layers is None:
        return list(range(MODEL_LAYER_COUNT))
    if not isinstance(layers, list) or not layers:
        raise RequestValidationError("activation_layers must be a non-empty list")
    if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers):
        raise RequestValidationError("activation_layers must contain integers")
    if len(layers) != len(set(layers)):
        raise RequestValidationError("activation_layers must not contain duplicates")
    invalid = [layer for layer in layers if not 0 <= layer < MODEL_LAYER_COUNT]
    if invalid:
        raise RequestValidationError(
            f"activation layers out of range for Qwen3-32B: {invalid}"
        )
    return sorted(layers)


def validate_generation_request(payload: Any) -> dict[str, Any]:
    """Validate and normalize one model-turn request without using compute."""

    if not isinstance(payload, dict):
        raise RequestValidationError("request must be a JSON object")

    trajectory_id = _require_string(payload, "trajectory_id")
    agent_id = _require_string(payload, "agent_id")
    agent_role = _require_string(payload, "agent_role")
    _validate_safe_id(trajectory_id, "trajectory_id")
    _validate_safe_id(agent_id, "agent_id")
    if agent_role not in ALLOWED_AGENT_ROLES:
        raise RequestValidationError(
            f"agent_role must be one of {sorted(ALLOWED_AGENT_ROLES)}"
        )

    step_index = payload.get("step_index")
    hop_index = payload.get("hop_index")
    for field, value in (("step_index", step_index), ("hop_index", hop_index)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RequestValidationError(f"{field} must be a non-negative integer")

    thinking_mode = _require_string(payload, "thinking_mode").lower()
    if thinking_mode not in THINKING_MODES:
        raise RequestValidationError("thinking_mode must be 'on' or 'off'")

    model_id = str(payload.get("model_id") or MODEL_ID)
    if model_id != MODEL_ID:
        raise RequestValidationError(f"model_id must be {MODEL_ID!r}")

    raw_poison_exposed = payload.get("raw_poison_exposed", False)
    if not isinstance(raw_poison_exposed, bool):
        raise RequestValidationError("raw_poison_exposed must be a boolean")
    if raw_poison_exposed and agent_id != RAW_POISON_AGENT_ID:
        raise RequestValidationError(
            "only worker_1 may receive the raw poisoned document"
        )
    if agent_id in {"worker_2", "executor_1"} and raw_poison_exposed:
        raise RequestValidationError(
            f"{agent_id} must receive only the upstream agent message"
        )

    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise RequestValidationError("tools must be a list")

    extract_activations = payload.get("extract_activations", True)
    if not isinstance(extract_activations, bool):
        raise RequestValidationError("extract_activations must be a boolean")

    return {
        "trajectory_id": trajectory_id,
        "step_index": step_index,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "hop_index": hop_index,
        "thinking_mode": thinking_mode,
        "enable_thinking": THINKING_MODES[thinking_mode],
        "model_id": model_id,
        "model_revision": str(payload.get("model_revision") or "main"),
        "messages": _validate_messages(payload.get("messages")),
        "tools": copy.deepcopy(tools),
        "raw_poison_exposed": raw_poison_exposed,
        "generation_settings": _validate_settings(payload.get("generation_settings")),
        "extract_activations": extract_activations,
        "activation_layers": _validate_layers(payload.get("activation_layers")),
        "activation_token_position": "last_generated_non_special_token",
    }


def split_thinking_text(text: str) -> dict[str, Any]:
    """Separate Qwen thinking content from the downstream-visible answer."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    opening = "<think>"
    closing = "</think>"
    if closing in text:
        before, after = text.rsplit(closing, 1)
        thinking = before.split(opening, 1)[-1].strip()
        return {
            "thinking_content": thinking,
            "final_content": after.strip(),
            "thinking_complete": True,
        }
    if opening in text:
        return {
            "thinking_content": text.split(opening, 1)[1].strip(),
            "final_content": "",
            "thinking_complete": False,
        }
    return {
        "thinking_content": None,
        "final_content": text.strip(),
        "thinking_complete": None,
    }


def activation_artifact_path(request: dict[str, Any]) -> str:
    """Return a safe path inside the shared activation Volume."""

    trajectory_id = str(request["trajectory_id"])
    thinking_mode = str(request["thinking_mode"])
    step_index = int(request["step_index"])
    _validate_safe_id(trajectory_id, "trajectory_id")
    path = PurePosixPath(
        "activations",
        trajectory_id,
        thinking_mode,
        f"step_{step_index:03d}.pt",
    )
    return path.as_posix()
