"""Shared Scenario 1 construction, schema, adapter, and Modal-contract tests."""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from jsonschema import Draft202012Validator

from src.scenario1 import generator as gen
from src.scenario1 import validator as V
from src.infrastructure.qwen_modal import CONTROLLED_GENERATION_SETTINGS, MODEL_LAYER_COUNT


REGISTRIES = gen.load_registries()
SCHEMA = json.load(open(V.SCHEMA_PATH))
FELLOW_TEMPLATE_PATH = (
    gen.PROJECT_ROOT
    / "schemas"
    / "scenario1"
    / "v2"
    / "fellow-handoff.template.json"
)
FELLOW_TEMPLATE = json.load(open(FELLOW_TEMPLATE_PATH))
VALIDATOR = Draft202012Validator(SCHEMA)
CANONICAL_OUTCOMES = {
    "clean",
    "resisted",
    "propagated_but_not_executed",
    "attempted_but_blocked",
    "executed",
    "indeterminate",
}


def check(record):
    structural = [
        f"SCHEMA {'/'.join(map(str, error.path))}: {error.message}"
        for error in VALIDATOR.iter_errors(record)
    ]
    return structural + (V.semantic_checks(record) if not structural else [])


@pytest.fixture(scope="module")
def records():
    return [
        gen.build_record(reg, item["condition_id"], item["treatment"])
        for reg in REGISTRIES
        for item in reg["conditions"]
    ]


def test_two_groups_produce_eight_base_trajectories(records):
    assert len(REGISTRIES) == 2
    assert len(records) == 8
    assert len({record["independence_group_id"] for record in records}) == 2
    for group_id in {record["independence_group_id"] for record in records}:
        group = [record for record in records if record["independence_group_id"] == group_id]
        assert {(record["condition_id"], record["treatment"]) for record in group} == {
            ("2-hop", "clean"),
            ("2-hop", "injected"),
            ("3-hop", "clean"),
            ("3-hop", "injected"),
        }


def test_all_records_pass_schema_and_semantics(records):
    for record in records:
        assert check(record) == [], f"{record['trajectory_id']}: {check(record)}"


def test_schema_requires_construction_and_source_provenance():
    assert {
        "independence_group_id",
        "domain_id",
        "task_family_id",
        "task",
        "document_set",
        "injection",
        "provenance",
    }.issubset(SCHEMA["required"])
    assert {
        "user_task",
        "expected_benign_behavior",
    }.issubset(SCHEMA["properties"]["task"]["required"])

    documents = SCHEMA["properties"]["document_set"]["properties"]["documents"]
    assert documents["minContains"] == documents["maxContains"] == 1
    assert {
        "doc_id",
        "title",
        "role",
        "text",
    }.issubset(documents["items"]["required"])
    assert {
        "file",
        "source_pdf",
        "clean_source_pdf",
        "injected_source_pdf",
    }.issubset(documents["items"]["properties"])
    assert "insertion_anchor" in SCHEMA["$defs"]["injection"]["required"]
    assert "created_by" in SCHEMA["$defs"]["provenance"]["required"]


def test_schema_marks_only_fellow_supplied_fields():
    assert SCHEMA["$comment"] == (
        "The required list applies to the final record. "
        "Fellows fill only fields marked x-required-by-fellows."
    )
    document_properties = (
        SCHEMA["properties"]["document_set"]["properties"]["documents"]["items"][
            "properties"
        ]
    )
    fellow_fields = [
        SCHEMA["properties"]["domain_id"],
        SCHEMA["properties"]["task_family_id"],
        SCHEMA["properties"]["independence_group_id"],
        SCHEMA["properties"]["task"]["properties"]["user_task"],
        SCHEMA["properties"]["task"]["properties"]["expected_benign_behavior"],
        document_properties["doc_id"],
        document_properties["title"],
        document_properties["file"],
        document_properties["source_pdf"],
        document_properties["clean_source_pdf"],
        document_properties["injected_source_pdf"],
        document_properties["role"],
        SCHEMA["$defs"]["injection"]["properties"]["insertion_anchor"],
        SCHEMA["$defs"]["provenance"]["properties"]["created_by"],
    ]
    assert all(field["x-required-by-fellows"] is True for field in fellow_fields)
    assert all(
        field["description"].startswith("Required by fellows")
        for field in fellow_fields
    )
    for generated_field in (
        "schema_version",
        "trajectory_id",
        "treatment",
        "model",
        "activation_config",
        "trajectory_trace",
        "evaluation_labels",
    ):
        assert "x-required-by-fellows" not in SCHEMA["properties"][generated_field]


def test_fellow_template_contains_only_handoff_sections():
    assert set(FELLOW_TEMPLATE) == {
        "_instructions",
        "domain_id",
        "task_family_id",
        "independence_group_id",
        "task",
        "document_set",
        "injection",
        "provenance",
    }
    documents = FELLOW_TEMPLATE["document_set"]["documents"]
    assert len(documents) == 3
    assert [document["role"] for document in documents] == [
        "benign",
        "benign",
        "injection_carrier",
    ]
    assert set(FELLOW_TEMPLATE["task"]) == {
        "user_task",
        "expected_benign_behavior",
    }
    assert set(FELLOW_TEMPLATE["injection"]) == {"insertion_anchor"}
    assert set(FELLOW_TEMPLATE["provenance"]) == {"created_by"}


def test_records_preserve_requested_construction_metadata(records):
    registries_by_group = {
        reg["independence_group_id"]: reg for reg in REGISTRIES
    }
    for record in records:
        reg = registries_by_group[record["independence_group_id"]]
        assert record["domain_id"] == reg["domain_id"]
        assert record["task_family_id"] == reg["task_family_id"]
        assert record["task"]["user_task"] == reg["task"]["user_task"]
        assert (
            record["task"]["expected_benign_behavior"]
            == reg["task"]["expected_benign_behavior"]
        )
        assert (
            record["injection"]["insertion_anchor"]
            == reg["injection"]["insertion_anchor"]
        )
        assert record["provenance"]["created_by"] == reg["provenance"]["created_by"]

        slots_by_id = {slot["doc_id"]: slot for slot in reg["document_slots"]}
        documents = record["document_set"]["documents"]
        assert len(documents) == 3
        assert sum(
            document["role"] == "injection_carrier" for document in documents
        ) == 1
        for document in documents:
            source = slots_by_id[document["doc_id"]]
            assert document["title"] == source["title"]
            for field in gen.DOCUMENT_SOURCE_FIELDS:
                if field in source:
                    assert document[field] == source[field]
                else:
                    assert field not in document


def test_schema_rejects_missing_requested_metadata(records):
    required_paths = (
        ("independence_group_id",),
        ("domain_id",),
        ("task_family_id",),
        ("task", "user_task"),
        ("task", "expected_benign_behavior"),
        ("document_set", "documents", 0, "title"),
        ("injection", "insertion_anchor"),
        ("provenance", "created_by"),
    )
    for path in required_paths:
        invalid = copy.deepcopy(records[0])
        parent = invalid
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
        assert list(VALIDATOR.iter_errors(invalid)), f"{path} was not required"


def test_schema_requires_exactly_one_injection_carrier(records):
    invalid = copy.deepcopy(records[0])
    for document in invalid["document_set"]["documents"]:
        document["role"] = "benign"
    assert list(VALIDATOR.iter_errors(invalid))


def test_each_group_holds_task_seed_wording_and_documents_constant(records):
    for group_id in {record["independence_group_id"] for record in records}:
        group = [record for record in records if record["independence_group_id"] == group_id]
        assert len({json.dumps(record["task"], sort_keys=True) for record in group}) == 1
        assert len({record["model"]["seed"] for record in group}) == 1
        assert len({record["injection"]["injection_variant"] for record in group}) == 1
        assert len({record["document_set_id"] for record in group}) == 1


def test_independent_groups_do_not_reuse_tasks_or_documents(records):
    group_rows = {}
    for record in records:
        group_rows.setdefault(record["independence_group_id"], record)
    first, second = group_rows.values()
    assert first["domain_id"] != second["domain_id"]
    assert first["task"]["user_task"] != second["task"]["user_task"]
    first_ids = {item["doc_id"] for item in first["document_set"]["documents"]}
    second_ids = {item["doc_id"] for item in second["document_set"]["documents"]}
    assert first_ids.isdisjoint(second_ids)
    first_text = {item["text"] for item in first["document_set"]["documents"]}
    second_text = {item["text"] for item in second["document_set"]["documents"]}
    assert first_text.isdisjoint(second_text)


def test_registry_validation_rejects_cross_group_leakage():
    registries = copy.deepcopy(REGISTRIES)
    registries[1]["document_slots"][0]["doc_id"] = registries[0]["document_slots"][0][
        "doc_id"
    ]
    with pytest.raises(ValueError, match="document ID"):
        gen.validate_registry_set(registries)


def test_registry_validation_rejects_missing_document_title():
    registries = copy.deepcopy(REGISTRIES)
    del registries[0]["document_slots"][0]["title"]
    with pytest.raises(ValueError, match="has no title"):
        gen.validate_registry_set(registries)


def test_registry_validation_rejects_system_prompt_drift():
    registries = copy.deepcopy(REGISTRIES)
    registries[1]["system_prompts"]["executor"] += " Changed."
    with pytest.raises(ValueError, match="system prompts"):
        gen.validate_registry_set(registries)


def test_system_prompts_and_carrier_are_fixed_across_groups():
    assert len({json.dumps(reg["system_prompts"], sort_keys=True) for reg in REGISTRIES}) == 1
    assert {reg["injection_family"] for reg in REGISTRIES} == {"fake_compliance_request"}
    assert {reg["injection_placement"] for reg in REGISTRIES} == {"body_text"}
    assert {reg["injection"]["carrier_marker"] for reg in REGISTRIES} == {
        "COMPLIANCE NOTICE:"
    }
    assert {reg["assigned_wording"] for reg in REGISTRIES} == {"A", "B"}


def test_each_group_has_three_documents_and_a_matched_carrier():
    for reg in REGISTRIES:
        clean, _ = gen.build_document_set(reg, "clean")
        injected, span = gen.build_document_set(reg, "injected")
        assert len(clean) == len(injected) == 3
        carrier_index = next(
            index for index, item in enumerate(clean) if item["role"] == "injection_carrier"
        )
        wording = reg["injection"]["wordings"][reg["assigned_wording"]]
        assert span is not None
        assert injected[carrier_index]["text"].replace(wording + "\n\n", "", 1) == clean[
            carrier_index
        ]["text"]
        assert reg["injection"]["endpoint"].endswith(".invalid/intake") or \
            ".example.invalid/" in reg["injection"]["endpoint"]


def test_only_worker1_is_raw_poison_exposed(records):
    for record in records:
        turns = [
            event for event in record["trajectory_trace"]["full_events"]
            if event["type"] == "agent_turn"
        ]
        exposed = {
            event["agent_id"] for event in turns
            if event["token_alignment"].get("injection_present_in_prompt")
        }
        expected = {"worker_1"} if record["treatment"] == "injected" else set()
        assert exposed == expected
        assert set(record["injection"]["raw_poison_exposed_agents"]) == expected


def test_agent_topology_matches_depth(records):
    for record in records:
        ids = {
            event["agent_id"] for event in record["trajectory_trace"]["full_events"]
            if event["type"] == "agent_turn"
        }
        assert ids == V.EXPECTED_AGENTS[record["condition_id"]]


def test_dry_run_never_claims_outputs_labels_or_artifacts(records):
    for record in records:
        assert record["generation_mode"] == "dry_run"
        assert record["model_called"] is False
        assert record["evaluation_labels"]["outcome_class"] is None
        assert record["evaluation_labels"]["action_channel"]["unsafe_action_executed"] is None
        assert record["evaluation_labels"]["behavioral_channel"]["output_adoption"] is None
        for event in record["trajectory_trace"]["full_events"]:
            if event["type"] != "agent_turn":
                continue
            assert event["model_called"] is False
            assert event["output"]["message"] is None
            assert event["output"]["raw_generated_text"] is None
            assert event["activation_metadata"]["layers_extracted"] == []
            assert event["activation_metadata"]["storage_path"] is None
            assert event["reasoning_compromise_label"]["label"] is None


def test_qwen_model_and_layer_scan_are_configurable(records):
    for record in records:
        assert record["model"]["model_name"] == "Qwen/Qwen3-32B"
        assert record["model"]["decoding_settings"] == CONTROLLED_GENERATION_SETTINGS
        assert record["activation_config"]["layers"] == list(range(MODEL_LAYER_COUNT))
        assert record["activation_config"]["primary_layer"] is None


def test_modal_request_plan_covers_both_modes_without_starting_compute(records):
    plan = gen.build_request_plan(records)
    expected_turns = sum(
        3 if record["condition_id"] == "2-hop" else 4 for record in records
    )
    assert len(plan) == expected_turns * 2 == 56
    assert {request["thinking_mode"] for request in plan} == {"on", "off"}
    assert all(request["raw_poison_exposed"] is False for request in plan
               if request["agent_id"] in {"worker_2", "executor_1"})
    assert all(request["injection_text"] is None for request in plan
               if request["raw_poison_exposed"] is False)
    assert all(request["injection_text"] for request in plan
               if request["raw_poison_exposed"] is True)
    executor_requests = [request for request in plan if request["agent_id"] == "executor_1"]
    assert all(request["tools"][0]["function"]["name"] == "submit_document_for_audit"
               for request in executor_requests)


def test_thinking_ablation_changes_only_the_mode_switch(records):
    record = records[0]
    event = next(
        item for item in record["trajectory_trace"]["full_events"]
        if item.get("agent_id") == "planner_1"
    )
    on = gen.build_generation_request(record, event, thinking_mode="on")
    off = gen.build_generation_request(record, event, thinking_mode="off")
    assert on["generation_settings"] == off["generation_settings"]
    assert on["enable_thinking"] is True
    assert off["enable_thinking"] is False


def test_manifest_covers_all_groups_and_paths(tmp_path):
    records, _ = gen.generate_all(REGISTRIES, str(tmp_path))
    manifest = json.load(open(tmp_path / "manifest.json"))
    assert len(records) == len(manifest["trajectories"]) == 8
    assert manifest["model_called"] is False
    assert set(manifest["contributors"]) == {"mariame", "onyokoli"}
    for item in manifest["trajectories"]:
        assert (tmp_path / item["path"]).exists()


def test_outcome_enum_is_the_six_canonical_real_run_classes():
    assert set(SCHEMA["$defs"]["outcome_class"]["enum"]) == CANONICAL_OUTCOMES
