"""Cross-PR tests for the shared generator, Modal result, and probe adapter."""

import copy
import json
from pathlib import Path

import pytest

from src.scenario1 import generator as scenario
from src.infrastructure.qwen_modal import generation_result_to_agent_turn_fields
from src.pipeline.handoff import normalize_handoff_json


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def base_records():
    return [
        scenario.build_record(reg, item["condition_id"], item["treatment"])
        for reg in scenario.load_registries()
        for item in reg["conditions"]
    ]


def test_all_shared_records_normalize_to_the_expected_probe_topology(base_records):
    expected = {
        "2-hop": ["planner", "worker", "executor"],
        "3-hop": ["planner", "worker", "worker2", "executor"],
    }
    for record in base_records:
        normalized = normalize_handoff_json(record)
        assert [item["node_id"] for item in normalized] == expected[record["condition_id"]]
        assert all(item["source_schema_version"] == "spec_gap.scenario1.v2"
                   for item in normalized)
        assert normalized[-1]["behavioral_outcome"] == "indeterminate"


def test_shared_retrieve_documents_event_reaches_probe_adapter(base_records):
    record = next(item for item in base_records if item["condition_id"] == "2-hop")
    normalized = normalize_handoff_json(record)
    worker = next(item for item in normalized if item["node_id"] == "worker")
    assert worker["tool_calls"][0]["tool"] == "retrieve_papers"
    assert len(worker["tool_calls"][0]["output"]["documents"]) == 3


def test_modal_result_fields_survive_the_shared_schema_and_probe_adapter(base_records):
    record = copy.deepcopy(next(
        item for item in base_records
        if item["condition_id"] == "3-hop" and item["treatment"] == "injected"
    ))
    modal_result = json.loads(
        (FIXTURE_DIR / "qwen_agent_turn_result.json").read_text()
    )
    fields = generation_result_to_agent_turn_fields(modal_result)
    worker = next(
        event for event in record["trajectory_trace"]["full_events"]
        if event.get("agent_id") == "worker_1"
    )
    worker.update(fields)
    worker["model_called"] = True
    worker["generation_mode"] = "real"
    record["generation_mode"] = "real"
    record["model_called"] = True

    normalized = normalize_handoff_json(record)
    normalized_worker = next(item for item in normalized if item["node_id"] == "worker")
    assert normalized_worker["rendered_prompt"] == modal_result["rendered_input"]
    assert normalized_worker["input_token_ids"] == modal_result["input_token_ids"]
    assert normalized_worker["generated_token_ids"] == modal_result["generated_token_ids"]
    assert normalized_worker["thinking_content"] == modal_result["thinking_content"]
    assert normalized_worker["tool_call_requests"] == modal_result["tool_call_requests"]
    assert normalized_worker["model_execution_metadata"]["thinking_mode"] == "on"


def test_hidden_thinking_is_not_used_as_the_downstream_message(base_records):
    record = copy.deepcopy(base_records[0])
    modal_result = json.loads(
        (FIXTURE_DIR / "qwen_agent_turn_result.json").read_text()
    )
    fields = generation_result_to_agent_turn_fields(modal_result)
    planner = next(
        event for event in record["trajectory_trace"]["full_events"]
        if event.get("agent_id") == "planner_1"
    )
    planner.update(fields)
    assert planner["output"]["message"] == modal_result["final_content"]
    assert planner["output"]["message"] != modal_result["raw_generated_text"]


def test_modal_tool_request_is_not_silently_recorded_as_executed(base_records):
    record = copy.deepcopy(next(
        item for item in base_records if item["condition_id"] == "2-hop"
    ))
    modal_result = json.loads(
        (FIXTURE_DIR / "qwen_agent_turn_result.json").read_text()
    )
    fields = generation_result_to_agent_turn_fields(modal_result)
    executor = next(
        event for event in record["trajectory_trace"]["full_events"]
        if event.get("agent_id") == "executor_1"
    )
    executor.update(fields)
    normalized = normalize_handoff_json(record)
    terminal = normalized[-1]
    assert terminal["tool_call_requests"][0]["status"] == "requested"
    assert terminal["tool_calls"] == []
    assert terminal["action_fired"] is None
