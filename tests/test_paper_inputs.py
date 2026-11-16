"""Tests for the fail-closed Scenario 1 paper-input policy."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.analysis.paper_inputs import (
    PAPER_INPUT_AUDIT_SCHEMA,
    load_paper_input_policy,
    select_paper_trajectory_records,
    validate_embedded_paper_input_audit,
    validate_paper_analysis_inputs,
)
from src.extraction.saved_activations import load_activation_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "experiments/scenario1/paper_input_policy.json"
INDEX_SCRIPT = (
    PROJECT_ROOT / "scripts/03_probe_analysis/07_build_activation_index.py"
)


def _trajectory(
    *,
    treatment: str,
    depth: str,
    mode: str,
    protocol: str,
    domain_id: str = "fin",
) -> dict:
    protocol_suffix = (
        "__gen_controlled_v2_5000"
        if protocol == "controlled_v2_5000"
        else ""
    )
    trajectory_id = (
        f"{domain_id}__{depth.replace('-', '')}__{treatment}"
        f"__prompt_neutral_v1{protocol_suffix}__thinking_{mode}"
    )
    record = {
        "trajectory_id": trajectory_id,
        "domain_id": domain_id,
        "agent_prompt_profile_id": "neutral_v1",
        "treatment": treatment,
        "delegation_depth": depth,
        "independence_group_id": domain_id,
        "matched_pair_id": (
            f"{domain_id}__{depth.replace('-', '')}__prompt_neutral_v1"
            f"{protocol_suffix}"
        ),
        "task_family_id": "scenario1",
        "document_set_id": f"{domain_id}__documents",
        "model": {
            "model_name": "Qwen/Qwen3-32B",
            "model_revision": "revision",
            "thinking_mode": mode,
        },
        "evaluation_labels": {
            "injection_present": treatment == "injected",
            "action_channel": {"unsafe_action_executed": False},
            "behavioral_channel": {"output_adoption": False},
            "outcome_class": "resisted" if treatment == "injected" else "clean",
        },
        "trajectory_trace": {
            "full_events": [{
                "type": "agent_turn",
                "step_index": 1,
                "agent_id": "worker_1",
                "agent_role": "worker",
                "hop_index": 1,
                # Keep both synthetic protocols in one selected tier here to
                # verify that the paper policy still rejects a misfiled v1.
                "model_execution_metadata": {"analysis_tier": "definitive"},
                "input": {
                    "rendered_prompt_hash": f"prompt-{trajectory_id}",
                    "input_token_ids": [1, 2, 3],
                },
                "output": {
                    "finish_reason": "stop",
                    "truncated": False,
                    "thinking_complete": True if mode == "on" else None,
                    "generated_token_ids": [4, 5],
                },
                "activation_metadata": {
                    "storage_status": "materialized",
                    "storage_path": (
                        f"activations/definitive/{trajectory_id}/{mode}/step_001.pt"
                    ),
                    "artifact_format_version": "spec_gap.activation_positions.v1",
                    "layers_extracted": [0, 1],
                    "checkpoint_positions": [{
                        "name": "last_input_token",
                        "sequence_index": 2,
                        "token_id": 3,
                    }],
                    "checkpoint_shapes": {"last_input_token": [2, 4]},
                    "checkpoint_forward_scopes": {
                        "last_input_token": "prompt_only"
                    },
                    "layer_metadata": {"dtype": "torch.bfloat16"},
                },
            }]
        },
    }
    if protocol != "controlled_v1_2048":
        record["generation_protocol_id"] = protocol
    return record


def _finance_matrix(protocol: str) -> list[dict]:
    return [
        _trajectory(
            treatment=treatment,
            depth=depth,
            mode=mode,
            protocol=protocol,
        )
        for treatment in ("clean", "injected")
        for depth in ("2-hop", "3-hop")
        for mode in ("off", "on")
    ]


def test_tracked_policy_points_to_complete_finance_v2_artifacts():
    policy = load_paper_input_policy(POLICY_PATH)
    finance = policy["domain_protocols"]["fin"]

    assert finance["required_generation_protocol_id"] == "controlled_v2_5000"
    assert finance["required_trajectory_id_fragment"] == (
        "__gen_controlled_v2_5000__"
    )
    # The policy is hash-bound provenance, so its recorded paths are frozen.
    # Pre-grid inputs were later moved under archive/old_experiment_inputs/.
    def _exists_now_or_archived(path: str) -> bool:
        archived = path.replace(
            "experiments/scenario1/inputs/", "archive/old_experiment_inputs/", 1
        )
        return (PROJECT_ROOT / path).is_file() or (PROJECT_ROOT / archived).is_file()

    assert all(
        _exists_now_or_archived(path)
        for path in finance["required_artifacts"].values()
    )


def test_selector_replaces_finance_v1_with_the_complete_v2_matrix():
    policy = load_paper_input_policy(POLICY_PATH)
    unmanaged = _trajectory(
        treatment="clean",
        depth="2-hop",
        mode="off",
        protocol="controlled_v1_2048",
        domain_id="aihc",
    )
    records = [*_finance_matrix("controlled_v1_2048"), unmanaged]
    records.extend(_finance_matrix("controlled_v2_5000"))

    selected, audit = select_paper_trajectory_records(records, policy)

    finance_ids = {
        record["trajectory_id"]
        for record in selected
        if record["domain_id"] == "fin"
    }
    assert len(finance_ids) == 8
    assert all("__gen_controlled_v2_5000__" in item for item in finance_ids)
    assert unmanaged in selected
    assert audit["schema_version"] == PAPER_INPUT_AUDIT_SCHEMA
    assert audit["controlled_domains"]["fin"] == {
        "status": "complete",
        "required_generation_protocol_id": "controlled_v2_5000",
        "required_agent_prompt_profile_id": "neutral_v1",
        "observed_generation_protocol_ids": [
            "controlled_v1_2048",
            "controlled_v2_5000",
        ],
        "selected_generation_protocol_ids": ["controlled_v2_5000"],
        "selected_trajectory_count": 8,
        "excluded_trajectory_count": 8,
    }
    assert len(audit["excluded_trajectory_ids"]) == 8
    validate_paper_analysis_inputs(selected, policy)


def test_downstream_validation_rejects_v1_or_an_incomplete_v2_matrix():
    policy = load_paper_input_policy(POLICY_PATH)

    with pytest.raises(ValueError, match="non-definitive trajectories"):
        validate_paper_analysis_inputs(
            _finance_matrix("controlled_v1_2048"),
            policy,
        )

    incomplete = _finance_matrix("controlled_v2_5000")[:-1]
    with pytest.raises(ValueError, match="exactly one trajectory per expected"):
        validate_paper_analysis_inputs(incomplete, policy)


def test_aggregate_audit_rejects_a_stale_policy_hash():
    policy = load_paper_input_policy(POLICY_PATH)
    selected, audit = select_paper_trajectory_records(
        _finance_matrix("controlled_v2_5000"),
        policy,
    )
    assert len(selected) == 8
    validate_embedded_paper_input_audit(
        {"paper_input_selection": audit},
        policy,
    )

    stale = json.loads(json.dumps(audit))
    stale["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash is stale"):
        validate_embedded_paper_input_audit(
            {"paper_input_selection": stale},
            policy,
        )


def test_activation_index_cli_filters_v1_and_records_policy_audit(tmp_path):
    trajectory_root = tmp_path / "trajectories"
    for record in [
        *_finance_matrix("controlled_v1_2048"),
        *_finance_matrix("controlled_v2_5000"),
    ]:
        mode_dir = (
            trajectory_root / "definitive" / record["model"]["thinking_mode"]
        )
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / f"{record['trajectory_id']}.json").write_text(
            json.dumps(record),
            encoding="utf-8",
        )

    output = tmp_path / "paper_activation_index.jsonl"
    summary_output = tmp_path / "paper_activation_index_summary.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(INDEX_SCRIPT),
            "--trajectory-root",
            str(trajectory_root),
            "--artifact-root",
            str(tmp_path),
            "--analysis-tier",
            "definitive",
            "--output",
            str(output),
            "--summary-output",
            str(summary_output),
            "--paper-input-policy",
            str(POLICY_PATH),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    rows = load_activation_index(output)
    assert len({row["trajectory_id"] for row in rows}) == 8
    assert all(
        "__gen_controlled_v2_5000__" in row["trajectory_id"] for row in rows
    )
    audit = validate_paper_analysis_inputs(
        rows,
        load_paper_input_policy(POLICY_PATH),
    )
    assert audit["controlled_domains"]["fin"]["status"] == "complete"

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["paper_input_selection"]["excluded_trajectory_count"] == 8
    assert json.loads(completed.stdout)["trajectories"] == 8
