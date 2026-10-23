"""Behavioral backend for OpenAI-compatible hosted endpoints (e.g. Modal GLM-5.3).

This backend is black-box: it returns visible text, optional reasoning text,
native tool calls, and token usage. It cannot return input token ids,
rendered prompts, or residual-stream activations, so its results are NOT
``qwen_modal`` generation results and must never be mixed into the white-box
probe cohort. ``src/scenario1/behavioral_orchestrator.py`` consumes them.

Configuration (environment):
  MODAL_PROXY_TOKEN   ``wk-<id>.ws-<secret>`` Modal proxy token (Bearer auth)
  SPEC_GAP_ENDPOINT_BASE_URL   default ``https://inference.us-west.modal.direct/v1``
  SPEC_GAP_ENDPOINT_MODEL      default ``zai-org/GLM-5.3``
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from src.infrastructure.qwen_modal import (
    parse_tool_call_requests,
    split_thinking_text,
)


DEFAULT_BASE_URL = "https://inference.us-west.modal.direct/v1"
DEFAULT_MODEL = "zai-org/GLM-5.3"
# Modal shared-endpoint list prices, USD per million tokens (2026-10).
DEFAULT_PRICING_PER_MTOK = {"prompt": 1.40, "cached_prompt": 0.26, "completion": 4.40}
BEHAVIORAL_RESULT_FORMAT = "spec_gap.behavioral_endpoint_turn.v1"

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


class EndpointError(RuntimeError):
    """Raised when the hosted endpoint fails or violates the run contract."""


def _urllib_transport(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:2000]
        raise EndpointError(f"HTTP {error.code}: {detail}") from error


@dataclass
class EndpointConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    token: str | None = None
    # Thinking-mode mapping. Modal's GLM-5.3 shared endpoint exposes reasoning
    # only as ``reasoning_effort`` in {low, high, max} and ignores
    # ``enable_thinking``; there is no true "off". So thinking_mode "off" is
    # run as the lowest effort and recorded as such. Set
    # ``reasoning_effort_off=None`` for servers that honour enable_thinking.
    reasoning_effort_on: str = "high"
    reasoning_effort_off: str | None = "low"
    # The endpoint advertises only temperature/top_p/stop/max_tokens. seed,
    # top_k and min_p are sent only when this is True.
    extended_sampling: bool = False
    # Reasoning tokens count against max_tokens. None keeps the protocol cap.
    max_tokens_override: int | None = None
    timeout_s: float = 600.0
    max_retries: int = 4
    pricing_per_mtok: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PRICING_PER_MTOK)
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> "EndpointConfig":
        config = cls(
            base_url=os.environ.get("SPEC_GAP_ENDPOINT_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("SPEC_GAP_ENDPOINT_MODEL", DEFAULT_MODEL),
            token=os.environ.get("MODAL_PROXY_TOKEN"),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(config, key, value)
        return config


def effective_reasoning_effort(thinking_mode: str, config: EndpointConfig) -> str | None:
    return config.reasoning_effort_on if thinking_mode == "on" else config.reasoning_effort_off


def build_chat_body(request: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
    """Map a validated SPEC-GAP generation request onto a chat-completions body.

    Messages, tools, temperature, top_p, seed and the token cap are carried over
    unchanged from the Qwen protocol so the behavioral matrix stays comparable.
    """

    settings = request["generation_settings"]
    thinking = request["thinking_mode"] == "on"
    body: dict[str, Any] = {
        "model": config.model,
        "messages": copy.deepcopy(request["messages"]),
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_tokens": config.max_tokens_override or settings["max_new_tokens"],
    }
    effort = effective_reasoning_effort(request["thinking_mode"], config)
    if effort is not None:
        body["reasoning_effort"] = effort
    else:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
        body["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if config.extended_sampling:
        body["seed"] = settings["seed"]
        body["top_k"] = settings.get("top_k")
        body["min_p"] = settings.get("min_p")
    if request["tools"]:
        body["tools"] = copy.deepcopy(request["tools"])
        body["tool_choice"] = "auto"
    return body


def _native_tool_calls(
    message: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = {
        tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)
    }
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, raw in enumerate(message.get("tool_calls") or []):
        function = (raw or {}).get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                errors.append({
                    "code": "invalid_tool_call_json",
                    "index": index,
                    "message": str(exc),
                    "raw": function.get("arguments"),
                })
                continue
        if not isinstance(name, str) or not name.strip():
            errors.append({"code": "missing_tool_name", "index": index, "raw": raw})
            continue
        if not isinstance(arguments, dict):
            errors.append({
                "code": "tool_arguments_must_be_object",
                "index": index,
                "name": name,
                "raw": raw,
            })
            continue
        calls.append({
            "name": name.strip(),
            "arguments": copy.deepcopy(arguments),
            "recognized": name.strip() in allowed,
            "status": "requested",
            "channel": "native_tool_call",
        })
    return calls, errors


def cost_usd(usage: dict[str, Any], pricing: dict[str, float]) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    )
    uncached = max(prompt - cached, 0)
    return (
        uncached * pricing["prompt"]
        + cached * pricing["cached_prompt"]
        + completion * pricing["completion"]
    ) / 1_000_000


def parse_chat_response(
    request: dict[str, Any],
    body: dict[str, Any],
    response: dict[str, Any],
    config: EndpointConfig,
    *,
    latency_s: float,
) -> dict[str, Any]:
    """Turn one chat-completions response into a behavioral turn result."""

    choices = response.get("choices") or []
    if not choices:
        raise EndpointError(f"endpoint returned no choices: {str(response)[:500]}")
    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")

    # Some servers inline <think> blocks instead of a reasoning field.
    split = split_thinking_text(content)
    final_content = split["final_content"]
    if reasoning is None and split["thinking_content"] is not None:
        reasoning = split["thinking_content"]

    native_calls, native_errors = _native_tool_calls(message, request["tools"])
    text_parsed = parse_tool_call_requests(final_content, request["tools"])
    for call in text_parsed["calls"]:
        call["channel"] = "text_tool_call_block"
    calls = native_calls + text_parsed["calls"]

    finish_reason = choice.get("finish_reason") or "unknown"
    usage = response.get("usage") or {}
    effort = effective_reasoning_effort(request["thinking_mode"], config)
    # Reasoning under "off" is only a violation when we asked for no thinking
    # at all; under an explicit low effort it is expected and recorded.
    thinking_leak = (
        request["thinking_mode"] == "off"
        and effort is None
        and bool((reasoning or "").strip())
    )
    return {
        "result_format": BEHAVIORAL_RESULT_FORMAT,
        "trajectory_id": request["trajectory_id"],
        "step_index": request["step_index"],
        "agent_id": request["agent_id"],
        "agent_role": request["agent_role"],
        "hop_index": request["hop_index"],
        "thinking_mode": request["thinking_mode"],
        "analysis_tier": request["analysis_tier"],
        "model_name": response.get("model") or config.model,
        "requested_model": config.model,
        "endpoint_base_url": config.base_url,
        "request_body_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "request_body_settings": {
            key: copy.deepcopy(value)
            for key, value in body.items()
            if key not in {"messages", "tools"}
        },
        "generation_settings": copy.deepcopy(request["generation_settings"]),
        "input_messages": copy.deepcopy(request["messages"]),
        "tools": copy.deepcopy(request["tools"]),
        "raw_poison_exposed": request["raw_poison_exposed"],
        "injection_text": request["injection_text"],
        "raw_content": content,
        "thinking_content": reasoning,
        "thinking_mode_violation": thinking_leak,
        "effective_reasoning_effort": effort,
        "reasoning_chars": len(reasoning or ""),
        "final_content": final_content,
        "downstream_message": final_content,
        "tool_call_requests": calls,
        "tool_call_parse_errors": native_errors + text_parsed["errors"],
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "usage": copy.deepcopy(usage),
        "cost_usd": cost_usd(usage, config.pricing_per_mtok),
        "latency_s": round(latency_s, 3),
        "response_id": response.get("id"),
        "system_fingerprint": response.get("system_fingerprint"),
        "activation_metadata": None,
    }


class EndpointClient:
    """Callable ``generate_turn`` backend for the behavioral orchestrator."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport or _urllib_transport
        self.sleep = sleep
        self.spent_usd = 0.0

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.config.token and self.transport is _urllib_transport:
            raise EndpointError("MODAL_PROXY_TOKEN is not set")
        body = build_chat_body(request, self.config)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.token}",
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            start = time.monotonic()
            try:
                response = self.transport(url, body, headers, self.config.timeout_s)
            except EndpointError as error:
                message = str(error)
                # Unsupported optional sampling params: drop them once, record it.
                if (
                    message.startswith("HTTP 400")
                    and self.config.extended_sampling
                    and any(k in message for k in ("top_k", "min_p", "seed"))
                ):
                    self.config.extended_sampling = False
                    body = build_chat_body(request, self.config)
                    last_error = error
                    continue
                retryable = any(
                    message.startswith(f"HTTP {code}")
                    for code in (408, 409, 429, 500, 502, 503, 504)
                )
                if not retryable:
                    raise
                last_error = error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = error
            else:
                result = parse_chat_response(
                    request,
                    body,
                    response,
                    self.config,
                    latency_s=time.monotonic() - start,
                )
                self.spent_usd += result["cost_usd"]
                return result
            self.sleep(min(60.0, 2.0 ** attempt * 5))
        raise EndpointError(f"endpoint failed after retries: {last_error}")
