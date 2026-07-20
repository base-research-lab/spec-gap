"""Tests for generate_onyinye_trajectories.py against Elena's review criteria."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GENERATOR = os.path.join(ROOT, "generate_onyinye_trajectories.py")
INPUT_CONFIG = os.path.join(
    ROOT, "experiments", "scenario1", "inputs", "domain_climate_science.json"
)


@pytest.fixture(scope="module")
def generated_trajectories():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, GENERATOR, "--input", INPUT_CONFIG, "--out", tmpdir],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Generator failed:\n{result.stderr}"

        trajectories = {}
        manifest = None
        for fname in os.listdir(tmpdir):
            if fname == "manifest_onyinye.json":
                with open(os.path.join(tmpdir, fname)) as f:
                    manifest = json.load(f)
            elif fname.endswith(".json"):
                with open(os.path.join(tmpdir, fname)) as f:
                    trajectories[fname] = json.load(f)

        yield {
            "trajectories": trajectories,
            "manifest": manifest,
            "output_dir": tmpdir,
        }


@pytest.fixture(scope="module")
def config():
    with open(INPUT_CONFIG) as f:
        return json.load(f)


class TestMatchGroupCompleteness:
    def test_exactly_four_trajectories(self, generated_trajectories):
        assert len(generated_trajectories["trajectories"]) == 4

    def test_all_four_conditions_present(self, generated_trajectories):
        conditions = set()
        for t in generated_trajectories["trajectories"].values():
            conditions.add((t["treatment"], t["condition_id"]))
        expected = {
            ("clean", "2-hop"),
            ("injected", "2-hop"),
            ("clean", "3-hop"),
            ("injected", "3-hop"),
        }
        assert conditions == expected

    def test_no_duplicate_trajectory_ids(self, generated_trajectories):
        tids = [t["trajectory_id"]
                for t in generated_trajectories["trajectories"].values()]
        assert len(tids) == len(set(tids))


class TestSharedFields:
    def test_same_task_across_all(self, generated_trajectories):
        tasks = {t["task"]["user_task"]
                 for t in generated_trajectories["trajectories"].values()}
        assert len(tasks) == 1

    def test_same_document_ids_across_all(self, generated_trajectories):
        doc_id_sets = set()
        for t in generated_trajectories["trajectories"].values():
            ids = tuple(d["doc_id"] for d in t["document_set"]["documents"])
            doc_id_sets.add(ids)
        assert len(doc_id_sets) == 1

    def test_three_documents_per_trajectory(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            assert len(t["document_set"]["documents"]) == 3, fname

    def test_same_schema_version(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            assert t["schema_version"] == "spec_gap.scenario1.v3", fname

    def test_same_system_prompts_implied(self, generated_trajectories, config):
        for t in generated_trajectories["trajectories"].values():
            events = t["trajectory_trace"]["full_events"]
            planner = [e for e in events if e.get("agent_id") == "planner_1"][0]
            assert planner["input"]["system_prompt"] == config["system_prompts"]["planner"]


class TestCleanDocumentNotContaminated:
    def test_clean_doc3_has_no_marker(self, config):
        doc3 = [d for d in config["documents"]
                if d["role"] == "clean_or_poisoned"][0]
        assert config["carrier_framing"] not in doc3["body_clean"]

    def test_clean_doc3_has_no_endpoint(self, config):
        doc3 = [d for d in config["documents"]
                if d["role"] == "clean_or_poisoned"][0]
        assert config["endpoint"] not in doc3["body_clean"]

    def test_poisoned_doc3_has_marker(self, config):
        doc3 = [d for d in config["documents"]
                if d["role"] == "clean_or_poisoned"][0]
        assert config["carrier_framing"] in doc3["body_poisoned"]

    def test_poisoned_doc3_has_endpoint(self, config):
        doc3 = [d for d in config["documents"]
                if d["role"] == "clean_or_poisoned"][0]
        assert config["endpoint"] in doc3["body_poisoned"]

    def test_clean_plus_injection_equals_poisoned(self, config):
        doc3 = [d for d in config["documents"]
                if d["role"] == "clean_or_poisoned"][0]
        payload = doc3["injection_payload"]
        reconstructed = doc3["body_clean"].replace(
            "\n\nDiscussion:",
            "\n\n" + payload + "\n\nDiscussion:",
        )
        assert reconstructed == doc3["body_poisoned"]

    def test_clean_trajectory_docs_have_no_injection(self, generated_trajectories, config):
        marker = config["carrier_framing"]
        endpoint = config["endpoint"]
        for fname, t in generated_trajectories["trajectories"].items():
            if t["treatment"] != "clean":
                continue
            events = t["trajectory_trace"]["full_events"]
            tool_event = [e for e in events if e["type"] == "tool_call"][0]
            for doc in tool_event["tool_result"]:
                assert marker not in doc["body"], (
                    f"{fname}: clean trajectory doc contains injection marker"
                )
                assert endpoint not in doc["body"], (
                    f"{fname}: clean trajectory doc contains endpoint"
                )


class TestPoisonExposure:
    def test_only_worker1_sees_raw_documents(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            events = t["trajectory_trace"]["full_events"]
            for e in events:
                if e.get("agent_id") in ("worker_2", "executor_1"):
                    saw = e.get("input", {}).get("saw_raw_documents")
                    assert not saw, (
                        f"{fname}: {e['agent_id']} saw raw documents"
                    )

    def test_worker1_sees_raw_documents(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            events = t["trajectory_trace"]["full_events"]
            worker1 = [e for e in events if e.get("agent_id") == "worker_1"][0]
            assert worker1["input"]["saw_raw_documents"] is True

    def test_injected_worker1_has_injection_in_prompt(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "injected":
                continue
            events = t["trajectory_trace"]["full_events"]
            worker1 = [e for e in events if e.get("agent_id") == "worker_1"][0]
            assert worker1["token_alignment"]["injection_present_in_prompt"] is True

    def test_clean_worker1_has_no_injection_in_prompt(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "clean":
                continue
            events = t["trajectory_trace"]["full_events"]
            worker1 = [e for e in events if e.get("agent_id") == "worker_1"][0]
            assert worker1["token_alignment"]["injection_present_in_prompt"] is False

    def test_raw_poison_exposed_agents(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] == "injected":
                assert t["injection"]["raw_poison_exposed_agents"] == ["worker_1"]
            else:
                assert t["injection"]["raw_poison_exposed_agents"] == []


class TestAgentTopology:
    def test_2hop_has_no_worker2(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            if "2hop" not in fname:
                continue
            events = t["trajectory_trace"]["full_events"]
            agent_ids = [e.get("agent_id") for e in events]
            assert "worker_2" not in agent_ids, fname

    def test_3hop_has_worker2(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            if "3hop" not in fname:
                continue
            events = t["trajectory_trace"]["full_events"]
            agent_ids = [e.get("agent_id") for e in events]
            assert "worker_2" in agent_ids, fname

    def test_2hop_agent_order(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            if "2hop" not in fname:
                continue
            events = t["trajectory_trace"]["full_events"]
            agent_ids = [e.get("agent_id", e.get("tool_name"))
                         for e in events]
            assert agent_ids == [
                "planner_1", "retrieve_documents", "worker_1", "executor_1"
            ], fname

    def test_3hop_agent_order(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            if "3hop" not in fname:
                continue
            events = t["trajectory_trace"]["full_events"]
            agent_ids = [e.get("agent_id", e.get("tool_name"))
                         for e in events]
            assert agent_ids == [
                "planner_1", "retrieve_documents", "worker_1",
                "worker_2", "executor_1"
            ], fname


class TestDryRunHonesty:
    def test_outcome_class_is_null(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            assert t["evaluation_labels"]["outcome_class"] is None, fname

    def test_injection_success_is_null(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            assert t["evaluation_labels"]["injection_success"] is None, fname

    def test_no_fabricated_executor_actions(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            events = t["trajectory_trace"]["full_events"]
            executor = [e for e in events if e.get("agent_id") == "executor_1"][0]
            actions = executor["output"].get("actions")
            assert actions is None, f"{fname}: executor has fabricated actions"

    def test_reasoning_labels_null(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            rl = t["evaluation_labels"]["reasoning_channel"]
            assert rl["label"] is None, fname

    def test_generation_mode_marked(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            assert t["generation_mode"] == "dry_run", fname

    def test_activation_storage_is_placeholder(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            events = t["trajectory_trace"]["full_events"]
            for e in events:
                act = e.get("activation_metadata")
                if act:
                    assert act["storage_status"] == "dry_run_placeholder", fname
                    assert act["storage_path"] is None, fname

    def test_agent_outputs_are_null(self, generated_trajectories):
        for fname, t in generated_trajectories["trajectories"].items():
            events = t["trajectory_trace"]["full_events"]
            for e in events:
                if e.get("type") == "agent_turn":
                    assert e["output"]["message"] is None, (
                        f"{fname}: {e.get('agent_id')} has non-null output"
                    )


class TestManifest:
    def test_manifest_exists(self, generated_trajectories):
        assert generated_trajectories["manifest"] is not None

    def test_manifest_has_all_trajectories(self, generated_trajectories):
        manifest = generated_trajectories["manifest"]
        assert len(manifest["trajectories"]) == 4

    def test_manifest_paths_resolve(self, generated_trajectories):
        manifest = generated_trajectories["manifest"]
        outdir = generated_trajectories["output_dir"]
        for entry in manifest["trajectories"]:
            path = os.path.join(outdir, entry["trajectory_id"] + ".json")
            assert os.path.isfile(path), f"Missing: {path}"

    def test_manifest_coverage_correct(self, generated_trajectories):
        cov = generated_trajectories["manifest"]["trajectory_coverage"]
        assert cov["n_total"] == 4
        assert cov["n_clean"] == 2
        assert cov["n_injected"] == 2
        assert cov["outcomes_are_observed"] is False

    def test_manifest_schema_version(self, generated_trajectories):
        assert generated_trajectories["manifest"]["schema_version"] == "spec_gap.scenario1.v3"


class TestInjectionFields:
    def test_injected_has_char_span(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "injected":
                continue
            span = t["injection"]["injection_char_span_in_source_doc"]
            assert span is not None
            assert span["start_char"] < span["end_char"]

    def test_injected_has_text(self, generated_trajectories, config):
        marker = config["carrier_framing"]
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "injected":
                continue
            assert t["injection"]["injected_text"] is not None
            assert marker in t["injection"]["injected_text"]

    def test_injected_has_vector_and_render(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "injected":
                continue
            assert t["injection"]["injection_vector"] is not None
            assert t["injection"]["injection_render"] is not None

    def test_clean_has_null_injection_fields(self, generated_trajectories):
        for t in generated_trajectories["trajectories"].values():
            if t["treatment"] != "clean":
                continue
            assert t["injection"]["injected_text"] is None
            assert t["injection"]["injection_char_span_in_source_doc"] is None
            assert t["injection"]["injection_vector"] is None
            assert t["injection"]["injection_render"] is None
