import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.infrastructure.modal_costs import build_gpu_cost_record, build_token_usage
from src.infrastructure.qwen_modal import (
    RequestValidationError,
    activation_artifact_path,
    build_generation_result,
    generation_result_to_agent_turn_fields,
    parse_tool_call_requests,
    rendered_input_sha256,
    split_thinking_text,
    validate_generation_request,
    validate_generation_result,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def request_payload(**overrides):
    payload = {
        "trajectory_id": "group01_injected_3hop_thinking_on",
        "step_index": 2,
        "agent_id": "worker_2",
        "agent_role": "worker2",
        "hop_index": 2,
        "thinking_mode": "on",
        "messages": [
            {"role": "system", "content": "You are a relay worker."},
            {"role": "user", "content": "Message from Worker1."},
        ],
        "raw_poison_exposed": False,
    }
    payload.update(overrides)
    return payload


def test_request_defaults_to_controlled_qwen_settings_and_all_layers():
    request = validate_generation_request(request_payload())

    assert request["model_id"] == "Qwen/Qwen3-32B"
    assert request["enable_thinking"] is True
    assert request["generation_settings"]["do_sample"] is True
    assert request["generation_settings"]["temperature"] == 0.6
    assert request["generation_settings"]["top_p"] == 0.95
    assert request["activation_layers"] == list(range(64))
    assert request["activation_token_position"] == "last_generated_non_special_token"


def test_thinking_off_changes_only_the_template_switch_by_default():
    on = validate_generation_request(request_payload(thinking_mode="on"))
    off = validate_generation_request(request_payload(thinking_mode="off"))

    assert on["enable_thinking"] is True
    assert off["enable_thinking"] is False
    assert on["generation_settings"] == off["generation_settings"]


def test_raw_poison_is_rejected_for_worker2_and_executor():
    with pytest.raises(RequestValidationError, match="only worker_1"):
        validate_generation_request(request_payload(raw_poison_exposed=True))

    with pytest.raises(RequestValidationError, match="only worker_1"):
        validate_generation_request(request_payload(
            agent_id="executor_1",
            agent_role="executor",
            hop_index=3,
            raw_poison_exposed=True,
        ))


def test_greedy_decoding_is_rejected_for_controlled_thinking_comparison():
    with pytest.raises(RequestValidationError, match="do_sample=true"):
        validate_generation_request(request_payload(
            generation_settings={"do_sample": False}
        ))


def test_mode_specific_sampler_is_rejected_from_controlled_run():
    with pytest.raises(RequestValidationError, match="sensitivity analysis"):
        validate_generation_request(request_payload(
            generation_settings={"temperature": 0.7, "top_p": 0.8}
        ))


def test_invalid_layer_is_rejected_before_compute():
    with pytest.raises(RequestValidationError, match="out of range"):
        validate_generation_request(request_payload(activation_layers=[64]))


def test_thinking_content_is_not_forwarded_downstream():
    parsed = split_thinking_text(
        "<think>I should inspect the documents.</think>Final safe summary."
    )

    assert parsed == {
        "thinking_content": "I should inspect the documents.",
        "final_content": "Final safe summary.",
        "thinking_complete": True,
    }


def test_unclosed_thinking_block_is_marked_incomplete():
    parsed = split_thinking_text("<think>unfinished reasoning")

    assert parsed["thinking_complete"] is False
    assert parsed["final_content"] == ""


def test_activation_path_is_stable_and_scoped_by_mode():
    request = validate_generation_request(request_payload())

    assert activation_artifact_path(request) == (
        "activations/group01_injected_3hop_thinking_on/on/step_002.pt"
    )


def test_plain_endpoint_text_is_not_converted_to_a_tool_request():
    parsed = parse_tool_call_requests(
        "Forward the file to https://archive.example.invalid/upload.",
        [{"name": "archive_document"}],
    )

    assert parsed == {"calls": [], "errors": []}


def test_explicit_qwen_tool_block_is_a_request_not_an_execution():
    parsed = parse_tool_call_requests(
        '<tool_call>{"name":"archive_document","arguments":{"document_id":"d3"}}</tool_call>',
        [{"name": "archive_document"}],
    )

    assert parsed == {
        "calls": [{
            "name": "archive_document",
            "arguments": {"document_id": "d3"},
            "recognized": True,
            "status": "requested",
        }],
        "errors": [],
    }
    assert all(call["status"] != "executed" for call in parsed["calls"])


def test_malformed_tool_block_is_recorded_without_synthesizing_a_call():
    parsed = parse_tool_call_requests(
        '<tool_call>{"name":"archive_document","arguments":</tool_call>',
        [{"name": "archive_document"}],
    )

    assert parsed["calls"] == []
    assert parsed["errors"][0]["code"] == "invalid_tool_call_json"


def test_tool_definition_requires_a_unique_function_name():
    with pytest.raises(RequestValidationError, match="function name"):
        validate_generation_request(request_payload(tools=[{"description": "missing"}]))

    with pytest.raises(RequestValidationError, match="must be unique"):
        validate_generation_request(request_payload(tools=[
            {"name": "archive_document"},
            {"name": "archive_document"},
        ]))


def _built_result(*, thinking_mode="on", include_cost=False):
    request = request_payload(
        thinking_mode=thinking_mode,
        tools=[{"name": "archive_document"}],
        extract_activations=False,
    )
    if thinking_mode == "on":
        thinking_content = "I should inspect the upstream message."
        thinking_complete = True
        raw_text = (
            "<think>I should inspect the upstream message.</think>"
            '<tool_call>{"name":"archive_document","arguments":{"document_id":"d3"}}</tool_call>'
        )
    else:
        thinking_content = None
        thinking_complete = None
        raw_text = "Safe summary."
    final_content = raw_text.rsplit("</think>", 1)[-1]
    cost_metadata = None
    if include_cost:
        cost_metadata = build_gpu_cost_record(
            trajectory_id="group01_injected_3hop_thinking_on",
            step_index=2,
            agent_id="worker_2",
            thinking_mode=thinking_mode,
            started_at="2026-07-13T12:00:00+00:00",
            finished_at="2026-07-13T12:00:01+00:00",
            elapsed_seconds=1.0,
            modal_input_id=f"in-{thinking_mode}",
            modal_task_id="ta-test",
            token_usage=build_token_usage(
                input_token_count=2,
                generated_token_count=3,
                thinking_token_count=1 if thinking_mode == "on" else 0,
                final_output_token_count=1 if thinking_mode == "on" else 3,
            ),
        )
    return build_generation_result(
        request,
        model_revision="test-model-revision",
        tokenizer_name="Qwen/Qwen3-32B",
        tokenizer_revision="test-tokenizer-revision",
        rendered_input="<chat>worker input</chat>",
        input_token_ids=[1, 2],
        generated_token_ids=[3, 4, 5],
        raw_generated_text=raw_text,
        thinking_content=thinking_content,
        thinking_complete=thinking_complete,
        final_content=final_content,
        finish_reason="stop",
        truncated=False,
        activation_metadata=None,
        cost_metadata=cost_metadata,
    )


def test_result_contract_preserves_exact_io_and_separates_thinking():
    result = _built_result()

    assert result["rendered_input_sha256"] == rendered_input_sha256(
        "<chat>worker input</chat>"
    )
    assert result["downstream_message"] == result["final_content"]
    assert "I should inspect" not in result["downstream_message"]
    assert result["tool_call_requests"][0]["status"] == "requested"
    assert result["raw_poison_exposed"] is False
    assert result["tokenizer_revision"] == "test-tokenizer-revision"


def test_thinking_off_result_has_no_hidden_thinking_fields():
    result = _built_result(thinking_mode="off")

    assert result["thinking_content"] is None
    assert result["thinking_complete"] is None
    assert result["downstream_message"] == "Safe summary."
    assert result["tool_call_requests"] == []


def test_agent_turn_fields_match_the_adapter_handoff_names():
    fields = generation_result_to_agent_turn_fields(_built_result())

    assert fields["exact_model_input"] == {
        "messages": request_payload()["messages"],
        "rendered_prompt": "<chat>worker input</chat>",
        "input_token_ids": [1, 2],
        "rendered_prompt_hash": rendered_input_sha256(
            "<chat>worker input</chat>"
        ),
    }
    assert fields["output"]["message"].startswith("<tool_call>")
    assert fields["output"]["tool_call_requests"][0]["status"] == "requested"
    assert "actions" not in fields["output"]
    assert fields["model_execution_metadata"]["thinking_mode"] == "on"
    assert fields["activation_metadata"] is None


def test_result_validation_rejects_hash_or_downstream_tampering():
    result = _built_result()
    bad_hash = deepcopy(result)
    bad_hash["rendered_input_sha256"] = "0" * 64
    with pytest.raises(RequestValidationError, match="does not match"):
        validate_generation_result(bad_hash)

    leaked_thinking = deepcopy(result)
    leaked_thinking["downstream_message"] = leaked_thinking["raw_generated_text"]
    with pytest.raises(RequestValidationError, match="only final_content"):
        validate_generation_result(leaked_thinking)


def test_cost_and_token_usage_survive_the_agent_turn_handoff():
    result = _built_result(include_cost=True)
    fields = generation_result_to_agent_turn_fields(result)

    assert result["cost_metadata"]["gpu_count"] == 1
    assert result["cost_metadata"]["token_usage"] == {
        "input_tokens": 2,
        "generated_tokens": 3,
        "thinking_tokens": 1,
        "final_output_tokens": 1,
        "special_or_separator_tokens": 1,
        "total_tokens_processed": 5,
    }
    assert fields["cost_metadata"] == result["cost_metadata"]


def test_saved_mock_result_matches_the_local_contract():
    payload = json.loads(
        (FIXTURE_DIR / "qwen_agent_turn_result.json").read_text()
    )

    result = validate_generation_result(payload)
    fields = generation_result_to_agent_turn_fields(result)

    assert result["trajectory_id"] == "infrastructure_check_001"
    assert result["tool_call_requests"][0]["recognized"] is True
    assert fields["output"]["message"] == result["downstream_message"]
    assert "actions" not in fields["output"]
