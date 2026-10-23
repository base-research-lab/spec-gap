"""Behavioral endpoint backend (GLM-5.3 on Modal) with a fake transport."""

from __future__ import annotations

import json

import pytest

from src.infrastructure.openai_endpoint import (
    EndpointClient,
    EndpointConfig,
    EndpointError,
    build_chat_body,
)
from src.scenario1.batch import build_live_batch
from src.scenario1.behavioral_orchestrator import run_behavioral_trajectory
from src.scenario1.domain_grid import registry_paths_for_domains
from src.scenario1.generator import INPUTS, load_registries
from src.scenario1.orchestrator import resolve_expected_endpoint


def _plan(tmp_path, treatment, depth="2-hop", mode="off"):
    paths = [str(p) for p in registry_paths_for_domains(("aihc",), inputs_root=INPUTS)][:1]
    items = build_live_batch(load_registries(paths), thinking_modes=[mode],
                             output_root=tmp_path)
    return next(i for i in items
                if i["treatment"] == treatment and i["condition_id"] == depth)


def _fake(behaviour):
    calls = []

    def transport(url, body, headers, timeout):
        calls.append(body)
        role = body["messages"][0]["content"]
        message = {"role": "assistant", "content": "Summary of the documents."}
        if body.get("tools") and behaviour.get("endpoint"):
            message["content"] = ""
            message["tool_calls"] = [{
                "id": "c1", "type": "function",
                "function": {"name": "submit_document_for_audit",
                             "arguments": json.dumps({
                                 "document_id": behaviour["doc_id"],
                                 "destination": behaviour["endpoint"],
                                 "content": "full text"})}}]
        return {"id": "r", "model": "zai-org/GLM-5.3",
                "choices": [{"message": message, "finish_reason":
                             "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
    return transport, calls


def test_body_maps_protocol_and_thinking(tmp_path):
    item = _plan(tmp_path, "clean", mode="on")
    record = item["structural_record"]
    transport, calls = _fake({})
    client = EndpointClient(EndpointConfig(token="t"), transport=transport)
    out = run_behavioral_trajectory(record, thinking_mode="on", generate_turn=client,
                                    model_label="glm")
    body = calls[0]
    assert body["model"] == "zai-org/GLM-5.3"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == record["model"]["decoding_settings"]["max_new_tokens"]
    assert "tools" not in body  # only the executor gets the tool
    assert "tools" in calls[-1]
    assert out["evaluation_labels"]["outcome_class"] == "clean"
    assert out["white_box_available"] is False
    assert client.spent_usd == pytest.approx(3 * (1000 * 1.40 + 100 * 4.40) / 1e6)


def test_native_tool_call_to_controlled_endpoint_is_executed(tmp_path):
    item = _plan(tmp_path, "injected")
    record = item["structural_record"]
    endpoint = resolve_expected_endpoint(record)
    doc_id = record["document_set"]["documents"][0]["doc_id"]
    transport, _ = _fake({"endpoint": endpoint, "doc_id": doc_id})
    client = EndpointClient(EndpointConfig(token="t"), transport=transport)
    out = run_behavioral_trajectory(record, thinking_mode="off",
                                    generate_turn=client, model_label="glm")
    assert out["evaluation_labels"]["outcome_class"] == "executed"
    assert out["evaluation_labels"]["action_channel"]["unsafe_action_executed"]


def test_injected_resisted(tmp_path):
    record = _plan(tmp_path, "injected", depth="3-hop")["structural_record"]
    transport, calls = _fake({})
    out = run_behavioral_trajectory(
        record, thinking_mode="off",
        generate_turn=EndpointClient(EndpointConfig(token="t"), transport=transport),
        model_label="glm")
    assert len(calls) == 4
    assert record["injection"]["injected_text"] in calls[1]["messages"][1]["content"]
    assert out["evaluation_labels"]["outcome_class"] == "resisted"


def test_thinking_leak_flagged_and_top_k_fallback(tmp_path):
    record = _plan(tmp_path, "clean")["structural_record"]
    state = {"n": 0}

    def transport(url, body, headers, timeout):
        state["n"] += 1
        if "top_k" in body:
            raise EndpointError("HTTP 400: unsupported parameter top_k")
        return {"choices": [{"message": {"content": "ok", "reasoning_content": "hmm"},
                             "finish_reason": "stop"}], "usage": {}}
    client = EndpointClient(EndpointConfig(token="t"), transport=transport,
                            sleep=lambda s: None)
    out = run_behavioral_trajectory(record, thinking_mode="off",
                                    generate_turn=client, model_label="glm")
    assert out["behavioral_run_summary"]["thinking_mode_violations"] == 3
    assert client.config.send_top_k is False


def test_body_without_tools_has_no_tool_choice():
    request = {"generation_settings": {"temperature": 0.6, "top_p": 0.95, "seed": 0,
                                       "max_new_tokens": 10, "top_k": 20, "min_p": 0},
               "thinking_mode": "off", "messages": [{"role": "user", "content": "x"}],
               "tools": []}
    body = build_chat_body(request, EndpointConfig())
    assert "tool_choice" not in body and "reasoning_effort" not in body
