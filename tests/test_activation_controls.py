import hashlib

import pytest
import torch

from src.analysis.activation_controls import (
    activation_control_table,
    run_activation_control_audit,
)


def _save_artifact(path, *, last_input, last_visible):
    artifact = {
        "activations": last_visible,
        "layers": [0, 1],
        "artifact_format_version": "spec_gap.activation_positions.v1",
        "primary_checkpoint": "last_visible_answer_token",
        "position_activations": {
            "last_input_token": last_input,
            "last_visible_answer_token": last_visible,
        },
    }
    torch.save(artifact, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    path,
    checksum,
    *,
    mode,
    treatment,
    agent_id,
    agent_role,
    checkpoint,
    prompt_hash,
    input_hash,
    generated_hash,
):
    return {
        "trajectory_id": f"group-a__2hop__{treatment}__thinking_{mode}",
        "step_index": 1 if agent_id == "planner_1" else 2,
        "thinking_mode": mode,
        "matched_pair_id": "group-a__2hop",
        "match_group_id": "group-a",
        "delegation_depth": "2-hop",
        "treatment": treatment,
        "agent_id": agent_id,
        "agent_role": agent_role,
        "checkpoint": checkpoint,
        "checkpoint_forward_scope": (
            "prompt_only"
            if checkpoint == "last_input_token"
            else "generated_prefix"
        ),
        "rendered_prompt_hash": prompt_hash,
        "input_token_count": 3,
        "input_token_ids_sha256": input_hash,
        "generated_token_count": 2,
        "generated_token_ids_sha256": generated_hash,
        "local_path": path.as_posix(),
        "checksum_sha256": checksum,
    }


def _audit_rows(tmp_path, *, mismatch_planner_input=False):
    rows = []
    for mode in ("off", "on"):
        for treatment, offset in (("clean", 0.0), ("injected", 2.0)):
            planner_path = tmp_path / f"planner-{mode}-{treatment}.pt"
            planner_checksum = _save_artifact(
                planner_path,
                last_input=torch.tensor([
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]) + (0.25 if mismatch_planner_input and treatment == "injected" else 0),
                last_visible=torch.tensor([
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ]) + offset,
            )
            for checkpoint in ("last_input_token", "last_visible_answer_token"):
                rows.append(_row(
                    planner_path,
                    planner_checksum,
                    mode=mode,
                    treatment=treatment,
                    agent_id="planner_1",
                    agent_role="planner",
                    checkpoint=checkpoint,
                    prompt_hash="planner-prompt",
                    input_hash="planner-input",
                    generated_hash=f"planner-output-{treatment}",
                ))

            worker_path = tmp_path / f"worker-{mode}-{treatment}.pt"
            worker_checksum = _save_artifact(
                worker_path,
                last_input=torch.tensor([
                    [20.0, 21.0, 22.0],
                    [23.0, 24.0, 25.0],
                ]) + offset,
                last_visible=torch.tensor([
                    [30.0, 31.0, 32.0],
                    [33.0, 34.0, 35.0],
                ]) + offset,
            )
            rows.append(_row(
                worker_path,
                worker_checksum,
                mode=mode,
                treatment=treatment,
                agent_id="worker_1",
                agent_role="worker",
                checkpoint="last_input_token",
                prompt_hash=f"worker-prompt-{treatment}",
                input_hash=f"worker-input-{treatment}",
                generated_hash=f"worker-output-{treatment}",
            ))
    return rows


def test_control_audit_separates_strict_and_stochastic_controls(tmp_path):
    result = run_activation_control_audit(_audit_rows(tmp_path))

    strict = result["strict_planner_input_control"]
    assert strict["status"] == "passed"
    assert strict["blocks_layer_selection"] is False
    assert {item["thinking_mode"] for item in strict["mode_controls"]} == {
        "off",
        "on",
    }
    assert all(
        item["exact_activation_pair_count"] == 1
        for item in strict["mode_controls"]
    )

    stochastic = result["stochastic_planner_output_control"]
    assert stochastic["status"] == "descriptive_only"
    assert all(
        item["generated_token_match_count"] == 0
        for item in stochastic["mode_checkpoint_controls"]
    )
    worker = next(
        profile
        for profile in result["propagation_profiles"]
        if profile["thinking_mode"] == "off"
        and profile["agent_id"] == "worker_1"
        and profile["checkpoint"] == "last_input_token"
    )
    assert worker["input_token_match_count"] == 0
    assert worker["layer_summaries"]["0"]["mean_relative_l2_difference"] > 0
    assert len(activation_control_table(result)) == 12


def test_control_audit_rejects_legacy_input_scope(tmp_path):
    rows = _audit_rows(tmp_path)
    for row in rows:
        if row["agent_id"] == "planner_1" and row["checkpoint"] == "last_input_token":
            row["checkpoint_forward_scope"] = None

    result = run_activation_control_audit(rows)

    strict = result["strict_planner_input_control"]
    assert strict["status"] == "failed"
    assert all(
        control["prompt_only_extraction"] is False
        for control in strict["mode_controls"]
    )


def test_control_audit_fails_changed_planner_input_activation(tmp_path):
    result = run_activation_control_audit(
        _audit_rows(tmp_path, mismatch_planner_input=True)
    )

    strict = result["strict_planner_input_control"]
    assert strict["status"] == "failed"
    assert strict["blocks_layer_selection"] is True
    assert "strict_planner_input_control_failed" in result[
        "layer_selection_blockers"
    ]


def test_control_audit_rejects_missing_token_identity_metadata(tmp_path):
    rows = _audit_rows(tmp_path)
    rows[0].pop("input_token_ids_sha256")

    with pytest.raises(ValueError, match="missing required fields"):
        run_activation_control_audit(rows)
