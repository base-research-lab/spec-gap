import json

import pytest

from src.scenario1.batch import (
    build_live_batch,
    has_current_activation_format,
    load_completed_batch_item,
    trajectory_turn_cost,
)
from src.scenario1.generator import load_registries


def test_live_batch_contains_full_two_mode_matrix(tmp_path):
    plan = build_live_batch(load_registries(), output_root=tmp_path)

    assert len(plan) == 16
    assert len({item["trajectory_id"] for item in plan}) == 16
    assert {item["thinking_mode"] for item in plan} == {"off", "on"}
    assert sum(3 if item["condition_id"] == "2-hop" else 4 for item in plan) == 56


def test_live_batch_rejects_invalid_mode_list(tmp_path):
    registries = load_registries()
    with pytest.raises(ValueError, match="non-empty unique"):
        build_live_batch(registries, thinking_modes=["off", "off"], output_root=tmp_path)
    with pytest.raises(ValueError, match="only 'off' and 'on'"):
        build_live_batch(registries, thinking_modes=["sometimes"], output_root=tmp_path)


def test_resume_rejects_mismatched_existing_record(tmp_path):
    item = build_live_batch(
        load_registries(), thinking_modes=["off"], output_root=tmp_path
    )[0]
    item["output_path"].parent.mkdir(parents=True)
    item["output_path"].write_text(json.dumps({"trajectory_id": "wrong"}))

    with pytest.raises(ValueError, match="invalid trajectory"):
        load_completed_batch_item(item)


def test_missing_batch_item_is_pending(tmp_path):
    item = build_live_batch(
        load_registries(), thinking_modes=["off"], output_root=tmp_path
    )[0]
    assert load_completed_batch_item(item) is None


def test_current_activation_format_requires_positions_for_each_turn():
    current = {
        "trajectory_trace": {
            "full_events": [{
                "type": "agent_turn",
                "output": {
                    "thinking_content": "Reasoning",
                    "final_content": "Answer",
                },
                "activation_metadata": {
                    "artifact_format_version": "spec_gap.activation_positions.v1",
                    "checkpoint_positions": [
                        {"name": "last_input_token"},
                        {"name": "last_reasoning_token"},
                        {"name": "last_visible_answer_token"},
                    ],
                    "checkpoint_forward_scopes": {
                        "last_input_token": "prompt_only",
                        "last_reasoning_token": "generated_prefix",
                        "last_visible_answer_token": "generated_prefix",
                    },
                },
            }]
        }
    }
    assert has_current_activation_format(current) is True

    legacy = json.loads(json.dumps(current))
    legacy["trajectory_trace"]["full_events"][0]["activation_metadata"].pop(
        "artifact_format_version"
    )
    assert has_current_activation_format(legacy) is False

    legacy_scope = json.loads(json.dumps(current))
    legacy_scope["trajectory_trace"]["full_events"][0][
        "activation_metadata"
    ].pop("checkpoint_forward_scopes")
    assert has_current_activation_format(legacy_scope) is False

    wrong_scope = json.loads(json.dumps(current))
    wrong_scope["trajectory_trace"]["full_events"][0][
        "activation_metadata"
    ]["checkpoint_forward_scopes"]["last_input_token"] = "generated_prefix"
    assert has_current_activation_format(wrong_scope) is False

    missing_reasoning = json.loads(json.dumps(current))
    missing_reasoning["trajectory_trace"]["full_events"][0][
        "activation_metadata"
    ]["checkpoint_positions"] = [
        {"name": "last_input_token"},
        {"name": "last_visible_answer_token"},
    ]
    assert has_current_activation_format(missing_reasoning) is False


def test_trajectory_turn_cost_sums_agent_turns_only():
    record = {
        "trajectory_trace": {
            "full_events": [
                {"type": "agent_turn", "cost_metadata": {"estimated_h200_cost_usd": 0.1}},
                {"type": "unsafe_action", "cost_metadata": {"estimated_h200_cost_usd": 99}},
                {"type": "agent_turn", "cost_metadata": {"estimated_h200_cost_usd": 0.2}},
            ]
        }
    }
    assert trajectory_turn_cost(record) == pytest.approx(0.3)
