"""Scenario 1: schema + generator-output structural tests.

Exercises the real input builder and the dry-run generator, not a single
hand-written trajectory. Only needs jsonschema (declared in pyproject).
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from jsonschema import Draft202012Validator

import scenario1_pipeline as gen
import validate_trajectory as V

REG = gen.load_registry()
SCHEMA = json.load(open(V.SCHEMA_PATH))
VALIDATOR = Draft202012Validator(SCHEMA)

CANONICAL_OUTCOMES = {"clean", "resisted", "propagated_but_not_executed",
                      "attempted_but_blocked", "executed", "indeterminate"}


def check(t):
    struct = [f"SCHEMA {'/'.join(map(str, e.path))}: {e.message}" for e in VALIDATOR.iter_errors(t)]
    return struct + (V.semantic_checks(t) if not struct else [])


@pytest.fixture(scope="module")
def records():
    return [gen.build_record(REG, c["condition_id"], c["treatment"]) for c in REG["conditions"]]


def test_four_conditions(records):
    assert {(r["condition_id"], r["treatment"]) for r in records} == {
        ("2-hop", "clean"), ("2-hop", "injected"),
        ("3-hop", "clean"), ("3-hop", "injected")}


def test_all_pass_schema_and_semantics(records):
    for r in records:
        assert check(r) == [], f"{r['trajectory_id']}: {check(r)}"


def test_group_shares_task_seed_wording(records):
    assert len({r["independence_group_id"] for r in records}) == 1
    assert len({json.dumps(r["task"], sort_keys=True) for r in records}) == 1
    assert len({r["model"]["seed"] for r in records}) == 1
    assert len({r["injection"]["injection_variant"] for r in records}) == 1  # one wording


def test_exactly_three_document_slots():
    assert len(REG["document_slots"]) == 3
    clean, _ = gen.build_document_set(REG, "clean")
    assert len(clean) == 3


def test_clean_injected_doc3_differ_only_by_span():
    clean, _ = gen.build_document_set(REG, "clean")
    inj, _ = gen.build_document_set(REG, "injected")
    marker = REG["injection"]["carrier_marker"]
    endpoint = REG["injection"]["endpoint"]
    wording = REG["injection"]["wordings"][REG["assigned_wording"]]
    c3, i3 = clean[2]["text"], inj[2]["text"]
    assert marker not in c3 and endpoint not in c3
    assert marker in i3 and endpoint in i3
    assert i3.replace(wording + "\n\n", "", 1) == c3


def test_only_worker1_exposed(records):
    for r in records:
        if not r["injection"]["injection_present"]:
            continue
        turns = [e for e in r["trajectory_trace"]["full_events"] if e["type"] == "agent_turn"]
        exposed = {e["agent_id"] for e in turns
                   if e["token_alignment"].get("injection_present_in_prompt")}
        assert exposed == {"worker_1"}
        assert r["injection"]["raw_poison_exposed_agents"] == ["worker_1"]


def test_agents_match_depth(records):
    for r in records:
        ids = {e["agent_id"] for e in r["trajectory_trace"]["full_events"]
               if e["type"] == "agent_turn"}
        assert ids == V.EXPECTED_AGENTS[r["condition_id"]]


def test_injected_is_propagated_not_executed(records):
    for r in records:
        el = r["evaluation_labels"]
        if r["injection"]["injection_present"]:
            assert el["outcome_class"] == "propagated_but_not_executed"
            assert el["action_channel"]["unsafe_action_executed"] is False
        else:
            assert el["outcome_class"] == "clean"


def test_reasoning_labels_null(records):
    for r in records:
        for e in r["trajectory_trace"]["full_events"]:
            if e["type"] == "agent_turn":
                assert e["reasoning_compromise_label"]["label"] is None


def test_outcome_enum_is_six_canonical():
    assert set(SCHEMA["$defs"]["outcome_class"]["enum"]) == CANONICAL_OUTCOMES


def test_all_six_outcomes_are_schema_valid(records):
    base = next(r for r in records if r["injection"]["injection_present"])
    for oc in CANONICAL_OUTCOMES:
        r = copy.deepcopy(base)
        r["evaluation_labels"]["outcome_class"] = oc
        assert not list(VALIDATOR.iter_errors(r)), f"{oc} not schema-valid"


def test_no_duplicate_trajectory_ids(records):
    ids = [r["trajectory_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_manifest_paths_resolve(tmp_path):
    gen.generate(REG, str(tmp_path))
    manifest = json.load(open(tmp_path / "manifest.json"))
    assert len(manifest["trajectories"]) == 4
    for item in manifest["trajectories"]:
        assert (tmp_path / item["path"]).exists()
