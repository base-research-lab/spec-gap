import pytest

from src.infrastructure.qwen_modal import (
    RequestValidationError,
    activation_artifact_path,
    split_thinking_text,
    validate_generation_request,
)


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
