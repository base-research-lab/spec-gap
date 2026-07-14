import hashlib
import numpy as np
import pytest
import torch

from src.extraction.saved_activations import (
    ACTIVATION_INDEX_SCHEMA,
    build_activation_index,
    filter_activation_index,
    load_activation_checkpoint,
    load_activation_index,
    load_probe_activation_batch,
    summarize_activation_index,
    write_activation_index,
)


def _save_current_artifact(path, *, offset=0.0):
    positions = {
        "last_input_token": torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset,
        "last_visible_answer_token": (
            torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100 + offset
        ),
    }
    artifact = {
        "activations": positions["last_visible_answer_token"],
        "layers": [0, 4, 8],
        "token_position": "last_generated_non_special_token",
        "token_id": 13,
        "artifact_format_version": "spec_gap.activation_positions.v1",
        "primary_checkpoint": "last_visible_answer_token",
        "position_activations": positions,
        "checkpoint_positions": [
            {"name": "last_input_token", "sequence_index": 5, "token_id": 10},
            {
                "name": "last_visible_answer_token",
                "sequence_index": 9,
                "token_id": 13,
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(storage_path, checksum, *, trajectory_id="trajectory-clean", injected=False):
    treatment = "injected" if injected else "clean"
    return {
        "trajectory_id": trajectory_id,
        "model": {
            "model_name": "Qwen/Qwen3-32B",
            "model_revision": "revision",
            "thinking_mode": "off",
        },
        "delegation_depth": "2-hop",
        "treatment": treatment,
        "independence_group_id": "group-1",
        "matched_pair_id": "group-1__2hop",
        "domain_id": "public_health",
        "task_family_id": "health-summary",
        "document_set_id": "health-docs",
        "evaluation_labels": {
            "injection_present": injected,
            "action_channel": {"unsafe_action_executed": False},
            "behavioral_channel": {"output_adoption": False},
            "outcome_class": "resisted" if injected else "clean",
        },
        "trajectory_trace": {
            "full_events": [{
                "type": "agent_turn",
                "step_index": 2,
                "agent_id": "worker_1",
                "agent_role": "worker",
                "hop_index": 1,
                "input": {
                    "rendered_prompt_hash": "prompt-hash",
                    "input_token_ids": [1, 2, 3],
                },
                "output": {
                    "finish_reason": "stop",
                    "truncated": False,
                    "thinking_complete": None,
                    "generated_token_ids": [4, 5],
                },
                "activation_metadata": {
                    "storage_status": "materialized",
                    "storage_path": storage_path,
                    "artifact_format_version": "spec_gap.activation_positions.v1",
                    "layers_extracted": [0, 4, 8],
                    "checkpoint_positions": [
                        {
                            "name": "last_input_token",
                            "sequence_index": 5,
                            "token_id": 10,
                        },
                        {
                            "name": "last_visible_answer_token",
                            "sequence_index": 9,
                            "token_id": 13,
                        },
                    ],
                    "checkpoint_shapes": {
                        "last_input_token": [3, 4],
                        "last_visible_answer_token": [3, 4],
                    },
                    "checkpoint_forward_scopes": {
                        "last_input_token": "prompt_only",
                        "last_visible_answer_token": "generated_prefix",
                    },
                    "layer_metadata": {
                        "dtype": "torch.float32",
                        "checksum_sha256": checksum,
                    },
                },
            }]
        },
    }


def test_build_index_and_summary(tmp_path):
    artifact = tmp_path / "activations/clean/off/step_002.pt"
    checksum = _save_current_artifact(artifact)
    rows = build_activation_index(
        [_record("activations/clean/off/step_002.pt", checksum)],
        artifact_root=tmp_path,
        require_local=True,
        verify_checksums=True,
    )

    assert len(rows) == 2
    assert {row["checkpoint"] for row in rows} == {
        "last_input_token",
        "last_visible_answer_token",
    }
    assert all(row["schema_version"] == ACTIVATION_INDEX_SCHEMA for row in rows)
    assert all(row["labels"]["injection_present"] == 0 for row in rows)
    assert all(row["rendered_prompt_hash"] == "prompt-hash" for row in rows)
    assert all(row["input_token_count"] == 3 for row in rows)
    assert all(row["generated_token_count"] == 2 for row in rows)
    assert {
        row["checkpoint"]: row["checkpoint_forward_scope"] for row in rows
    } == {
        "last_input_token": "prompt_only",
        "last_visible_answer_token": "generated_prefix",
    }
    assert all(len(row["input_token_ids_sha256"]) == 64 for row in rows)
    assert all(len(row["generated_token_ids_sha256"]) == 64 for row in rows)
    summary = summarize_activation_index(rows)
    assert summary["trajectories"] == 1
    assert summary["model_turns"] == 1
    assert summary["local_artifacts"] == 1
    assert summary["checkpoints"]["last_input_token"] == 1


def test_index_round_trip_and_filter(tmp_path):
    artifact = tmp_path / "activations/clean/off/step_002.pt"
    checksum = _save_current_artifact(artifact)
    rows = build_activation_index(
        [_record("activations/clean/off/step_002.pt", checksum)],
        artifact_root=tmp_path,
    )
    index_path = tmp_path / "index.jsonl"
    write_activation_index(rows, index_path)

    loaded = load_activation_index(index_path)
    selected = filter_activation_index(
        loaded,
        thinking_mode="off",
        agent_role="worker",
        checkpoint="last_input_token",
    )
    assert len(selected) == 1
    assert selected[0]["step_index"] == 2


def test_load_named_checkpoint_and_layers(tmp_path):
    artifact = tmp_path / "activation.pt"
    _save_current_artifact(artifact)

    loaded = load_activation_checkpoint(
        artifact,
        checkpoint="last_input_token",
    )
    assert loaded.checkpoint == "last_input_token"
    assert loaded.layers == (0, 4, 8)
    assert loaded.activations.shape == (3, 4)
    assert loaded.activations[1].tolist() == [4.0, 5.0, 6.0, 7.0]


def test_legacy_artifact_supports_primary_only(tmp_path):
    artifact = tmp_path / "legacy.pt"
    torch.save({
        "activations": torch.ones(2, 3),
        "layers": [1, 2],
        "token_position": "last_generated_non_special_token",
        "token_id": 13,
    }, artifact)

    loaded = load_activation_checkpoint(artifact)
    assert loaded.checkpoint == "primary"
    with pytest.raises(ValueError, match="requires a current multi-position"):
        load_activation_checkpoint(artifact, checkpoint="last_visible_answer_token")


def test_probe_batch_aligns_layers_labels_and_metadata(tmp_path):
    clean_path = tmp_path / "activations/clean/off/step_002.pt"
    injected_path = tmp_path / "activations/injected/off/step_002.pt"
    clean_checksum = _save_current_artifact(clean_path)
    injected_checksum = _save_current_artifact(injected_path, offset=10.0)
    records = [
        _record("activations/clean/off/step_002.pt", clean_checksum),
        _record(
            "activations/injected/off/step_002.pt",
            injected_checksum,
            trajectory_id="trajectory-injected",
            injected=True,
        ),
    ]
    rows = filter_activation_index(
        build_activation_index(records, artifact_root=tmp_path, require_local=True),
        checkpoint="last_input_token",
    )

    batch = load_probe_activation_batch(
        rows,
        label_target="injection_present",
        layers=[4, 8],
    )
    assert batch.labels.tolist() == [0, 1]
    assert set(batch.activations) == {4, 8}
    assert batch.activations[4].shape == (2, 4)
    assert np.allclose(batch.activations[4][1] - batch.activations[4][0], 10.0)
    assert [row["trajectory_id"] for row in batch.metadata] == [
        "trajectory-clean",
        "trajectory-injected",
    ]


def test_missing_artifact_and_checksum_mismatch_fail_closed(tmp_path):
    missing_record = _record("activations/missing.pt", "abc")
    with pytest.raises(FileNotFoundError, match="not available locally"):
        build_activation_index(
            [missing_record],
            artifact_root=tmp_path,
            require_local=True,
        )

    artifact = tmp_path / "activation.pt"
    _save_current_artifact(artifact)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        load_activation_checkpoint(
            artifact,
            expected_checksum="0" * 64,
            verify_checksum=True,
        )


def test_index_rejects_path_outside_artifact_root(tmp_path):
    record = _record("../outside.pt", "abc")
    with pytest.raises(ValueError, match="escapes artifact_root"):
        build_activation_index([record], artifact_root=tmp_path)


def test_index_loader_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="contains no rows"):
        load_activation_index(path)
