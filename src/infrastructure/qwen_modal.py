"""Pure-Python validation helpers for the Qwen3-32B Modal runner.

This module does not import Modal, Torch, or Transformers. Scenario generators
can validate a request locally before any remote function or GPU is started.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

from .modal_costs import validate_gpu_cost_record


MODEL_ID = "Qwen/Qwen3-32B"
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
MODEL_LAYER_COUNT = 64
THINKING_MODES = {"on": True, "off": False}
ALLOWED_AGENT_ROLES = {"planner", "worker", "worker2", "executor"}
RAW_POISON_AGENT_ID = "worker_1"
ACTIVATION_ARTIFACT_FORMAT = "spec_gap.activation_positions.v1"
ACTIVATION_TOKEN_POSITION = "last_generated_non_special_token"
ACTIVATION_CHECKPOINT_NAMES = {
    "last_input_token",
    "last_reasoning_token",
    "last_visible_answer_token",
}

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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL,
)


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


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def _validate_tools(tools: Any) -> list[dict[str, Any]]:
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise RequestValidationError("tools must be a list")
    normalized = []
    names = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise RequestValidationError(f"tools[{index}] must be an object")
        name = _tool_name(tool)
        if name is None:
            raise RequestValidationError(
                f"tools[{index}] must define a non-empty function name"
            )
        _validate_safe_id(name, f"tools[{index}] function name")
        names.append(name)
        normalized.append(copy.deepcopy(tool))
    if len(names) != len(set(names)):
        raise RequestValidationError("tool function names must be unique")
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
    messages = _validate_messages(payload.get("messages"))
    injection_text = payload.get("injection_text")
    if raw_poison_exposed:
        if not isinstance(injection_text, str) or not injection_text:
            raise RequestValidationError(
                "raw-poison requests require a non-empty injection_text"
            )
        occurrences = sum(
            message["content"].count(injection_text)
            for message in messages
        )
        if occurrences != 1:
            raise RequestValidationError(
                "injection_text must appear exactly once in the model messages"
            )
    elif injection_text is not None:
        raise RequestValidationError(
            "injection_text must be null when raw_poison_exposed is false"
        )

    tools = _validate_tools(payload.get("tools", []))

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
        "model_revision": str(payload.get("model_revision") or MODEL_REVISION),
        "messages": messages,
        "tools": tools,
        "raw_poison_exposed": raw_poison_exposed,
        "injection_text": injection_text,
        "generation_settings": _validate_settings(payload.get("generation_settings")),
        "extract_activations": extract_activations,
        "activation_layers": _validate_layers(payload.get("activation_layers")),
        "activation_token_position": ACTIVATION_TOKEN_POSITION,
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


def build_activation_checkpoint_plan(
    *,
    input_token_ids: list[int],
    generated_token_ids: list[int],
    enable_thinking: bool,
    thinking_close_offset: int | None,
    special_token_ids: list[int] | set[int],
) -> dict[str, Any]:
    """Select comparable token positions without storing every token state."""

    input_ids = _validate_token_ids(input_token_ids, "input_token_ids")
    generated_ids = _validate_token_ids(
        generated_token_ids, "generated_token_ids"
    )
    if not isinstance(enable_thinking, bool):
        raise RequestValidationError("enable_thinking must be a boolean")
    if thinking_close_offset is not None and (
        not isinstance(thinking_close_offset, int)
        or isinstance(thinking_close_offset, bool)
        or not 0 <= thinking_close_offset < len(generated_ids)
    ):
        raise RequestValidationError("thinking_close_offset is out of range")
    if not enable_thinking and thinking_close_offset is not None:
        raise RequestValidationError(
            "thinking-off generations cannot define thinking_close_offset"
        )
    special = set(special_token_ids)
    if any(not isinstance(token_id, int) for token_id in special):
        raise RequestValidationError("special_token_ids must contain integers")

    def last_non_special_offset(
        token_ids: list[int], *, start_offset: int = 0
    ) -> int | None:
        for local_offset in range(len(token_ids) - 1, -1, -1):
            if token_ids[local_offset] not in special:
                return start_offset + local_offset
        return None

    checkpoints = [{
        "name": "last_input_token",
        "sequence_index": len(input_ids) - 1,
        "generated_token_offset": None,
        "token_id": input_ids[-1],
    }]
    if enable_thinking:
        reasoning_end = (
            thinking_close_offset
            if thinking_close_offset is not None
            else len(generated_ids)
        )
        reasoning_offset = last_non_special_offset(generated_ids[:reasoning_end])
        if reasoning_offset is not None:
            checkpoints.append({
                "name": "last_reasoning_token",
                "sequence_index": len(input_ids) + reasoning_offset,
                "generated_token_offset": reasoning_offset,
                "token_id": generated_ids[reasoning_offset],
            })
        visible_offset = None
        if thinking_close_offset is not None:
            visible_start = thinking_close_offset + 1
            visible_offset = last_non_special_offset(
                generated_ids[visible_start:], start_offset=visible_start
            )
    else:
        visible_offset = last_non_special_offset(generated_ids)

    if visible_offset is not None:
        checkpoints.append({
            "name": "last_visible_answer_token",
            "sequence_index": len(input_ids) + visible_offset,
            "generated_token_offset": visible_offset,
            "token_id": generated_ids[visible_offset],
        })

    primary_offset = last_non_special_offset(generated_ids)
    if primary_offset is None:
        primary_offset = len(generated_ids) - 1
    primary_sequence_index = len(input_ids) + primary_offset
    primary_checkpoint = next(
        (
            checkpoint["name"]
            for checkpoint in checkpoints
            if checkpoint["sequence_index"] == primary_sequence_index
        ),
        None,
    )
    if primary_checkpoint is None:
        raise RequestValidationError(
            "last generated token does not match an activation checkpoint"
        )
    return {
        "artifact_format_version": ACTIVATION_ARTIFACT_FORMAT,
        "primary_checkpoint": primary_checkpoint,
        "primary_sequence_index": primary_sequence_index,
        "checkpoints": checkpoints,
    }


def parse_tool_call_requests(
    final_content: str,
    tools: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Parse only Qwen's explicit Hermes-style tool-call blocks.

    Plain text, including an endpoint written in prose, is never converted into
    a tool request. A parsed request is still only a request; the shared
    executor must separately record whether an allowed simulated action fired.
    """

    if not isinstance(final_content, str):
        raise TypeError("final_content must be a string")
    normalized_tools = _validate_tools(tools)
    allowed_names = {_tool_name(tool) for tool in normalized_tools}
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    matches = list(_TOOL_CALL_BLOCK.finditer(final_content))

    if final_content.count("<tool_call>") != final_content.count("</tool_call>"):
        errors.append({
            "code": "unbalanced_tool_call_tags",
            "message": "tool-call tags are not balanced",
        })

    for index, match in enumerate(matches):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append({
                "code": "invalid_tool_call_json",
                "index": index,
                "message": str(exc),
                "raw": raw,
            })
            continue
        if not isinstance(parsed, dict):
            errors.append({
                "code": "tool_call_must_be_object",
                "index": index,
                "raw": raw,
            })
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if not isinstance(name, str) or not name.strip():
            errors.append({
                "code": "missing_tool_name",
                "index": index,
                "raw": raw,
            })
            continue
        if not isinstance(arguments, dict):
            errors.append({
                "code": "tool_arguments_must_be_object",
                "index": index,
                "name": name,
                "raw": raw,
            })
            continue
        name = name.strip()
        calls.append({
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "recognized": name in allowed_names,
            "status": "requested",
        })

    return {"calls": calls, "errors": errors}


def rendered_input_sha256(rendered_input: str) -> str:
    """Hash the exact rendered chat input saved with a model turn."""

    if not isinstance(rendered_input, str) or not rendered_input:
        raise ValueError("rendered_input must be a non-empty string")
    return hashlib.sha256(rendered_input.encode("utf-8")).hexdigest()


def build_injection_token_alignment(
    *,
    rendered_input: str,
    input_token_ids: list[int],
    injection_text: str | None,
    offset_mapping: list[tuple[int, int]] | None,
    tokenizer_name: str,
    tokenizer_revision: str,
) -> dict[str, Any]:
    """Map the injected text onto the exact rendered Qwen input.

    Character and token spans use an inclusive start and exclusive end. The
    caller obtains ``offset_mapping`` from the same tokenizer that produced the
    model input and verifies that its token IDs match ``input_token_ids``.
    """

    prompt_hash = rendered_input_sha256(rendered_input)
    base = {
        "injection_present_in_prompt": injection_text is not None,
        "char_span": None,
        "injection_token_span": None,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "rendered_prompt_hash": prompt_hash,
        "truncation_removed_injection_tokens": False,
        "span_convention": "start_inclusive_end_exclusive",
    }
    if injection_text is None:
        return base
    if rendered_input.count(injection_text) != 1:
        raise RequestValidationError(
            "injection_text must appear exactly once in rendered_input"
        )
    if offset_mapping is None or len(offset_mapping) != len(input_token_ids):
        raise RequestValidationError(
            "offset_mapping must match the exact input_token_ids length"
        )
    if any(
        not isinstance(offset, (list, tuple))
        or len(offset) != 2
        or any(not isinstance(value, int) for value in offset)
        or not 0 <= offset[0] <= offset[1] <= len(rendered_input)
        for offset in offset_mapping
    ):
        raise RequestValidationError("offset_mapping contains an invalid span")

    start_char = rendered_input.index(injection_text)
    end_char = start_char + len(injection_text)
    token_indices = [
        index
        for index, (start, end) in enumerate(offset_mapping)
        if end > start_char and start < end_char
    ]
    if not token_indices:
        raise RequestValidationError("no input tokens overlap injection_text")
    start_token = token_indices[0]
    end_token = token_indices[-1] + 1
    if token_indices != list(range(start_token, end_token)):
        raise RequestValidationError("injection token span is not contiguous")

    base["char_span"] = {
        "start_char": start_char,
        "end_char": end_char,
    }
    base["injection_token_span"] = {
        "start_token": start_token,
        "end_token": end_token,
    }
    return base


def build_generation_result(
    request: dict[str, Any],
    *,
    model_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    rendered_input: str,
    input_token_ids: list[int],
    generated_token_ids: list[int],
    raw_generated_text: str,
    thinking_content: str | None,
    thinking_complete: bool | None,
    final_content: str,
    finish_reason: str,
    truncated: bool,
    activation_metadata: dict[str, Any] | None,
    cost_metadata: dict[str, Any] | None = None,
    token_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the stable output of one Modal model turn."""

    normalized_request = validate_generation_request(request)
    if token_alignment is None:
        if normalized_request["injection_text"] is not None:
            raise RequestValidationError(
                "raw-poison generation results require token_alignment"
            )
        token_alignment = build_injection_token_alignment(
            rendered_input=rendered_input,
            input_token_ids=input_token_ids,
            injection_text=None,
            offset_mapping=None,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
        )
    parsed_tools = parse_tool_call_requests(final_content, normalized_request["tools"])
    result = {
        "trajectory_id": normalized_request["trajectory_id"],
        "step_index": normalized_request["step_index"],
        "agent_id": normalized_request["agent_id"],
        "agent_role": normalized_request["agent_role"],
        "hop_index": normalized_request["hop_index"],
        "model_name": MODEL_ID,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "thinking_mode": normalized_request["thinking_mode"],
        "enable_thinking": normalized_request["enable_thinking"],
        "generation_settings": copy.deepcopy(
            normalized_request["generation_settings"]
        ),
        "input_messages": copy.deepcopy(normalized_request["messages"]),
        "tools": copy.deepcopy(normalized_request["tools"]),
        "raw_poison_exposed": normalized_request["raw_poison_exposed"],
        "injection_text": normalized_request["injection_text"],
        "rendered_input": rendered_input,
        "rendered_input_sha256": rendered_input_sha256(rendered_input),
        "input_token_ids": list(input_token_ids),
        "generated_token_ids": list(generated_token_ids),
        "raw_generated_text": raw_generated_text,
        "thinking_content": thinking_content,
        "thinking_complete": thinking_complete,
        "final_content": final_content,
        "parsed_message": {
            "role": "assistant",
            "content": final_content,
            "tool_calls": copy.deepcopy(parsed_tools["calls"]),
        },
        "tool_call_requests": copy.deepcopy(parsed_tools["calls"]),
        "tool_call_parse_errors": copy.deepcopy(parsed_tools["errors"]),
        "downstream_message": final_content,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "token_alignment": copy.deepcopy(token_alignment),
        "activation_metadata": copy.deepcopy(activation_metadata),
        "cost_metadata": copy.deepcopy(cost_metadata),
    }
    return validate_generation_result(result)


def _validate_token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RequestValidationError(f"{field} must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise RequestValidationError(
            f"{field} must contain non-negative integers"
        )
    return list(value)


def _validate_result_token_alignment(
    result: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    alignment = result.get("token_alignment")
    if alignment is None and request["injection_text"] is None:
        return build_injection_token_alignment(
            rendered_input=result["rendered_input"],
            input_token_ids=result["input_token_ids"],
            injection_text=None,
            offset_mapping=None,
            tokenizer_name=result["tokenizer_name"],
            tokenizer_revision=result["tokenizer_revision"],
        )
    if not isinstance(alignment, dict):
        raise RequestValidationError("token_alignment must be an object")
    if alignment.get("injection_present_in_prompt") is not request[
        "raw_poison_exposed"
    ]:
        raise RequestValidationError(
            "token_alignment presence must match raw_poison_exposed"
        )
    if alignment.get("tokenizer_name") != result["tokenizer_name"]:
        raise RequestValidationError(
            "token_alignment.tokenizer_name must match tokenizer_name"
        )
    if alignment.get("tokenizer_revision") != result["tokenizer_revision"]:
        raise RequestValidationError(
            "token_alignment.tokenizer_revision must match tokenizer_revision"
        )
    if alignment.get("rendered_prompt_hash") != result["rendered_input_sha256"]:
        raise RequestValidationError(
            "token_alignment rendered prompt hash does not match"
        )
    if alignment.get("truncation_removed_injection_tokens") is not False:
        raise RequestValidationError(
            "live input must preserve every injection token"
        )
    if alignment.get("span_convention") != "start_inclusive_end_exclusive":
        raise RequestValidationError("token_alignment span convention is invalid")

    char_span = alignment.get("char_span")
    token_span = alignment.get("injection_token_span")
    if request["injection_text"] is None:
        if char_span is not None or token_span is not None:
            raise RequestValidationError(
                "non-exposed turns must not contain injection spans"
            )
        return copy.deepcopy(alignment)
    if not isinstance(char_span, dict) or not isinstance(token_span, dict):
        raise RequestValidationError(
            "raw-poison turns require character and token spans"
        )
    start_char = char_span.get("start_char")
    end_char = char_span.get("end_char")
    start_token = token_span.get("start_token")
    end_token = token_span.get("end_token")
    if (
        not isinstance(start_char, int)
        or not isinstance(end_char, int)
        or not 0 <= start_char < end_char <= len(result["rendered_input"])
        or result["rendered_input"][start_char:end_char]
        != request["injection_text"]
    ):
        raise RequestValidationError(
            "token_alignment character span does not select injection_text"
        )
    if (
        not isinstance(start_token, int)
        or not isinstance(end_token, int)
        or not 0 <= start_token < end_token <= len(result["input_token_ids"])
    ):
        raise RequestValidationError("token_alignment token span is invalid")
    return copy.deepcopy(alignment)


def _validate_activation_checkpoints(
    activation: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Validate the optional multi-position activation artifact contract."""

    format_version = activation.get("artifact_format_version")
    checkpoints = activation.get("checkpoint_positions")
    if format_version is None and checkpoints is None:
        return
    if format_version != ACTIVATION_ARTIFACT_FORMAT:
        raise RequestValidationError(
            f"activation artifact_format_version must be {ACTIVATION_ARTIFACT_FORMAT!r}"
        )
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RequestValidationError(
            "activation checkpoint_positions must be a non-empty list"
        )
    if any(not isinstance(checkpoint, dict) for checkpoint in checkpoints):
        raise RequestValidationError(
            "activation checkpoint_positions must contain objects"
        )
    names = [checkpoint.get("name") for checkpoint in checkpoints]
    if len(names) != len(set(names)) or any(
        name not in ACTIVATION_CHECKPOINT_NAMES for name in names
    ):
        raise RequestValidationError(
            "activation checkpoint names must be unique controlled positions"
        )
    if "last_input_token" not in names:
        raise RequestValidationError(
            "activation checkpoints must include last_input_token"
        )
    if result["thinking_mode"] == "off" and "last_reasoning_token" in names:
        raise RequestValidationError(
            "thinking-off activations cannot include last_reasoning_token"
        )
    if result["thinking_mode"] == "off" and "last_visible_answer_token" not in names:
        raise RequestValidationError(
            "thinking-off activations must include last_visible_answer_token"
        )
    if result["thinking_mode"] == "on" and result["thinking_content"]:
        if "last_reasoning_token" not in names:
            raise RequestValidationError(
                "thinking-on activations must include last_reasoning_token"
            )
    if result["final_content"] and "last_visible_answer_token" not in names:
        raise RequestValidationError(
            "visible output requires a last_visible_answer_token checkpoint"
        )

    all_token_ids = result["input_token_ids"] + result["generated_token_ids"]
    input_length = len(result["input_token_ids"])
    checkpoint_shapes = activation.get("checkpoint_shapes")
    if (
        not isinstance(checkpoint_shapes, dict)
        or set(checkpoint_shapes) != set(names)
    ):
        raise RequestValidationError(
            "activation checkpoint_shapes must cover every checkpoint"
        )
    checkpoint_forward_scopes = activation.get("checkpoint_forward_scopes")
    if checkpoint_forward_scopes is not None:
        if (
            not isinstance(checkpoint_forward_scopes, dict)
            or set(checkpoint_forward_scopes) != set(names)
        ):
            raise RequestValidationError(
                "activation checkpoint_forward_scopes must cover every checkpoint"
            )
        expected_scopes = {
            name: (
                "prompt_only" if name == "last_input_token" else "generated_prefix"
            )
            for name in names
        }
        if checkpoint_forward_scopes != expected_scopes:
            raise RequestValidationError(
                "activation checkpoint_forward_scopes are inconsistent"
            )
    for checkpoint in checkpoints:
        name = checkpoint["name"]
        index = checkpoint.get("sequence_index")
        offset = checkpoint.get("generated_token_offset")
        token_id = checkpoint.get("token_id")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(all_token_ids)
        ):
            raise RequestValidationError(
                f"activation checkpoint {name} has an invalid sequence_index"
            )
        if token_id != all_token_ids[index]:
            raise RequestValidationError(
                f"activation checkpoint {name} token_id does not match the sequence"
            )
        expected_offset = None if index < input_length else index - input_length
        if offset != expected_offset:
            raise RequestValidationError(
                f"activation checkpoint {name} generated_token_offset is inconsistent"
            )
        shape = checkpoint_shapes[name]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != len(activation["layers"])
            or any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 1
                for size in shape
            )
        ):
            raise RequestValidationError(
                f"activation checkpoint {name} has an invalid shape"
            )

    input_checkpoint = next(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint["name"] == "last_input_token"
    )
    if (
        input_checkpoint["sequence_index"] != input_length - 1
        or input_checkpoint["token_id"] != result["input_token_ids"][-1]
    ):
        raise RequestValidationError(
            "last_input_token checkpoint must identify the final prompt token"
        )
    primary = activation.get("primary_checkpoint")
    by_name = {checkpoint["name"]: checkpoint for checkpoint in checkpoints}
    if primary not in by_name:
        raise RequestValidationError(
            "activation primary_checkpoint must name a saved checkpoint"
        )
    if activation.get("token_id") != by_name[primary]["token_id"]:
        raise RequestValidationError(
            "activation primary token_id must match primary_checkpoint"
        )
    sequence_length = activation.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length != max(item["sequence_index"] for item in checkpoints) + 1
    ):
        raise RequestValidationError(
            "activation sequence_length must end at the primary checkpoint"
        )


def validate_generation_result(payload: Any) -> dict[str, Any]:
    """Validate a saved model-turn result without loading Modal or Qwen."""

    if not isinstance(payload, dict):
        raise RequestValidationError("generation result must be a JSON object")
    result = copy.deepcopy(payload)
    reconstructed_request = {
        "trajectory_id": result.get("trajectory_id"),
        "step_index": result.get("step_index"),
        "agent_id": result.get("agent_id"),
        "agent_role": result.get("agent_role"),
        "hop_index": result.get("hop_index"),
        "thinking_mode": result.get("thinking_mode"),
        "model_id": result.get("model_name"),
        "model_revision": result.get("model_revision"),
        "messages": result.get("input_messages"),
        "tools": result.get("tools"),
        "raw_poison_exposed": result.get("raw_poison_exposed"),
        "injection_text": result.get("injection_text"),
        "generation_settings": result.get("generation_settings"),
        "extract_activations": result.get("activation_metadata") is not None,
        "activation_layers": (
            (result.get("activation_metadata") or {}).get("layers")
            if result.get("activation_metadata") is not None
            else [0]
        ),
    }
    request = validate_generation_request(reconstructed_request)
    if result.get("enable_thinking") is not request["enable_thinking"]:
        raise RequestValidationError(
            "enable_thinking must match thinking_mode"
        )

    for field in ("model_revision", "tokenizer_name", "tokenizer_revision"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RequestValidationError(f"{field} must be a non-empty string")

    rendered_input = result.get("rendered_input")
    expected_hash = rendered_input_sha256(rendered_input)
    if result.get("rendered_input_sha256") != expected_hash:
        raise RequestValidationError("rendered_input_sha256 does not match rendered_input")
    result["input_token_ids"] = _validate_token_ids(
        result.get("input_token_ids"), "input_token_ids"
    )
    result["generated_token_ids"] = _validate_token_ids(
        result.get("generated_token_ids"), "generated_token_ids"
    )
    result["token_alignment"] = _validate_result_token_alignment(result, request)

    for field in ("raw_generated_text", "final_content", "downstream_message"):
        if not isinstance(result.get(field), str):
            raise RequestValidationError(f"{field} must be a string")
    if result["downstream_message"] != result["final_content"]:
        raise RequestValidationError(
            "downstream_message must contain only final_content"
        )
    if request["enable_thinking"]:
        if result.get("thinking_content") is not None and not isinstance(
            result.get("thinking_content"), str
        ):
            raise RequestValidationError("thinking_content must be a string or null")
        if not isinstance(result.get("thinking_complete"), bool):
            raise RequestValidationError(
                "thinking_complete must be a boolean when thinking is enabled"
            )
    elif result.get("thinking_content") is not None or result.get("thinking_complete") is not None:
        raise RequestValidationError(
            "thinking-off results must not contain hidden thinking content"
        )

    parsed_message = result.get("parsed_message")
    if not isinstance(parsed_message, dict):
        raise RequestValidationError("parsed_message must be an object")
    if parsed_message.get("role") != "assistant":
        raise RequestValidationError("parsed_message.role must be 'assistant'")
    if parsed_message.get("content") != result["final_content"]:
        raise RequestValidationError(
            "parsed_message.content must match final_content"
        )
    reparsed = parse_tool_call_requests(result["final_content"], request["tools"])
    if result.get("tool_call_requests") != reparsed["calls"]:
        raise RequestValidationError(
            "tool_call_requests do not match explicit tool-call blocks"
        )
    if result.get("tool_call_parse_errors") != reparsed["errors"]:
        raise RequestValidationError(
            "tool_call_parse_errors do not match explicit tool-call blocks"
        )
    if parsed_message.get("tool_calls") != reparsed["calls"]:
        raise RequestValidationError(
            "parsed_message.tool_calls must match tool_call_requests"
        )

    finish_reason = result.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise RequestValidationError("finish_reason must be 'stop' or 'length'")
    if not isinstance(result.get("truncated"), bool):
        raise RequestValidationError("truncated must be a boolean")
    if result["truncated"] != (finish_reason == "length"):
        raise RequestValidationError(
            "truncated must be true exactly when finish_reason is 'length'"
        )

    activation = result.get("activation_metadata")
    if activation is not None:
        if not isinstance(activation, dict):
            raise RequestValidationError("activation_metadata must be an object or null")
        if activation.get("storage_status") != "materialized":
            raise RequestValidationError(
                "a returned activation artifact must have storage_status='materialized'"
            )
        storage_path = activation.get("storage_path")
        expected_path = activation_artifact_path(request)
        if storage_path != expected_path:
            raise RequestValidationError(
                f"activation storage_path must be {expected_path!r}"
            )
        checksum = activation.get("checksum_sha256")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise RequestValidationError(
                "activation checksum_sha256 must be 64 lowercase hex characters"
            )
        if activation.get("layers") != request["activation_layers"]:
            raise RequestValidationError(
                "activation layers must match the generation request"
            )
        if activation.get("token_position") != request["activation_token_position"]:
            raise RequestValidationError(
                "activation token_position must match the generation request"
            )
        _validate_activation_checkpoints(activation, result)
    cost = result.get("cost_metadata")
    if cost is not None:
        try:
            cost = validate_gpu_cost_record(cost)
        except ValueError as exc:
            raise RequestValidationError(f"invalid cost_metadata: {exc}") from exc
        for field in ("trajectory_id", "step_index", "agent_id", "thinking_mode"):
            if cost.get(field) != result.get(field):
                raise RequestValidationError(
                    f"cost_metadata.{field} must match the model-turn result"
                )
        usage = cost["token_usage"]
        if usage["input_tokens"] != len(result["input_token_ids"]):
            raise RequestValidationError(
                "cost input token count must match input_token_ids"
            )
        if usage["generated_tokens"] != len(result["generated_token_ids"]):
            raise RequestValidationError(
                "cost generated token count must match generated_token_ids"
            )
        if result["thinking_mode"] == "off" and usage["thinking_tokens"] != 0:
            raise RequestValidationError(
                "thinking-off cost records must have zero thinking tokens"
            )
        result["cost_metadata"] = cost
    else:
        result["cost_metadata"] = None
    return result


def generation_result_to_agent_turn_fields(payload: Any) -> dict[str, Any]:
    """Return the model-owned fields the shared generator can add to an event."""

    result = validate_generation_result(payload)
    return {
        "token_alignment": copy.deepcopy(result["token_alignment"]),
        "exact_model_input": {
            "messages": copy.deepcopy(result["input_messages"]),
            "rendered_prompt": result["rendered_input"],
            "input_token_ids": list(result["input_token_ids"]),
            "rendered_prompt_hash": result["rendered_input_sha256"],
        },
        "output": {
            "message": result["final_content"],
            "raw_generated_text": result["raw_generated_text"],
            "generated_token_ids": list(result["generated_token_ids"]),
            "thinking_content": result["thinking_content"],
            "thinking_complete": result["thinking_complete"],
            "final_content": result["final_content"],
            "parsed_message": copy.deepcopy(result["parsed_message"]),
            "tool_call_requests": copy.deepcopy(result["tool_call_requests"]),
            "tool_call_parse_errors": copy.deepcopy(
                result["tool_call_parse_errors"]
            ),
            "finish_reason": result["finish_reason"],
            "truncated": result["truncated"],
        },
        "activation_metadata": copy.deepcopy(result["activation_metadata"]),
        "cost_metadata": copy.deepcopy(result["cost_metadata"]),
        "model_execution_metadata": {
            "model_name": result["model_name"],
            "model_revision": result["model_revision"],
            "tokenizer_name": result["tokenizer_name"],
            "tokenizer_revision": result["tokenizer_revision"],
            "thinking_mode": result["thinking_mode"],
            "enable_thinking": result["enable_thinking"],
            "generation_settings": copy.deepcopy(result["generation_settings"]),
            "raw_poison_exposed": result["raw_poison_exposed"],
        },
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
