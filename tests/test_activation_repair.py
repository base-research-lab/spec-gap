from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from src.infrastructure.modal_costs import build_gpu_cost_record, build_token_usage
from src.infrastructure.qwen_modal import (
    RequestValidationError,
    activation_artifact_path,
    build_generation_result,
    build_injection_token_alignment,
)
from src.scenario1.activation_repair import (
    ACTIVATION_REPAIR_METHOD,
    ACTIVATION_REPAIR_RESULT_SCHEMA,
    activation_repair_backup_path,
    apply_activation_repair_result,
    build_activation_repair_plan,
    reconcile_activation_repair_metadata,
    select_activation_repairs,
    summarize_activation_repair_plan,
    validate_activation_repair_request,
    validate_local_activation_repair_sources,
    verify_activation_repair_smoke_pair,
)
from src.scenario1.generator import build_record, load_registries
from src.scenario1.orchestrator import (
    run_live_trajectory,
    write_live_trajectory,
    write_model_turn_result,
)


def _legacy_result(request, final_content, checksum):
    input_ids = [1, 2, 3 + request["step_index"]]
    generated_ids = [100 + request["step_index"]]
    rendered_input = json.dumps(request["messages"], sort_keys=True)
    injection_text = request.get("injection_text")
    offset_mapping = None
    if injection_text is not None:
        start = rendered_input.index(injection_text)
        end = start + len(injection_text)
        offset_mapping = [(0, start), (start, end), (end, len(rendered_input))]
    alignment = build_injection_token_alignment(
        rendered_input=rendered_input,
        input_token_ids=input_ids,
        injection_text=injection_text,
        offset_mapping=offset_mapping,
        tokenizer_name=request["model_id"],
        tokenizer_revision=request["model_revision"],
    )
    checkpoints = [
        {
            "name": "last_input_token",
            "sequence_index": 2,
            "generated_token_offset": None,
            "token_id": input_ids[-1],
        },
        {
            "name": "last_visible_answer_token",
            "sequence_index": 3,
            "generated_token_offset": 0,
            "token_id": generated_ids[-1],
        },
    ]
    return build_generation_result(
        request,
        model_revision=request["model_revision"],
        tokenizer_name=request["model_id"],
        tokenizer_revision=request["model_revision"],
        rendered_input=rendered_input,
        input_token_ids=input_ids,
        generated_token_ids=generated_ids,
        raw_generated_text=final_content,
        thinking_content=None,
        thinking_complete=None,
        final_content=final_content,
        finish_reason="stop",
        truncated=False,
        activation_metadata={
            "storage_status": "materialized",
            "storage_path": activation_artifact_path(request),
            "checksum_sha256": checksum,
            "shape": [64, 4],
            "dtype": "torch.bfloat16",
            "layers": request["activation_layers"],
            "token_position": request["activation_token_position"],
            "token_id": generated_ids[-1],
            "artifact_format_version": "spec_gap.activation_positions.v1",
            "primary_checkpoint": "last_visible_answer_token",
            "checkpoint_positions": checkpoints,
            "checkpoint_shapes": {
                checkpoint["name"]: [64, 4] for checkpoint in checkpoints
            },
            "sequence_length": 4,
        },
        token_alignment=alignment,
    )


def _write_live_pair(tmp_path):
    output_root = tmp_path / "experiments/scenario1/trajectories"
    registry = next(
        item
        for item in load_registries()
        if item["independence_group_id"] == "climate_coastal_flood_01"
    )
    for treatment in ("clean", "injected"):
        structural = build_record(registry, "2-hop", treatment)

        def generate(request):
            old_bytes = _legacy_artifact_bytes(request)
            checksum = hashlib.sha256(old_bytes).hexdigest()
            artifact = tmp_path / activation_artifact_path(request)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(old_bytes)
            result = _legacy_result(
                request,
                f"Safe output from {request['agent_id']}.",
                checksum,
            )
            write_model_turn_result(result, output_root)
            return result

        live = run_live_trajectory(
            structural,
            thinking_mode="off",
            generate_turn=generate,
        )
        write_live_trajectory(live, output_root)
    return output_root


def _legacy_artifact_bytes(request):
    positions = {
        "last_input_token": torch.full(
            (64, 4),
            2.0 if "__injected__" in request["trajectory_id"] else 1.0,
            dtype=torch.bfloat16,
        ),
        "last_visible_answer_token": torch.full(
            (64, 4),
            float(request["step_index"] + 10),
            dtype=torch.bfloat16,
        ),
    }
    buffer = io.BytesIO()
    torch.save(
        {
            "activations": positions["last_visible_answer_token"],
            "layers": request["activation_layers"],
            "token_position": request["activation_token_position"],
            "token_id": 100 + request["step_index"],
            "artifact_format_version": "spec_gap.activation_positions.v1",
            "primary_checkpoint": "last_visible_answer_token",
            "position_activations": positions,
            "checkpoint_positions": [
                {
                    "name": "last_input_token",
                    "sequence_index": 2,
                    "generated_token_offset": None,
                    "token_id": 3 + request["step_index"],
                },
                {
                    "name": "last_visible_answer_token",
                    "sequence_index": 3,
                    "generated_token_offset": 0,
                    "token_id": 100 + request["step_index"],
                },
            ],
        },
        buffer,
    )
    return buffer.getvalue()


def _repair_metadata(request):
    return {
        "schema_version": ACTIVATION_REPAIR_RESULT_SCHEMA,
        "repair_method": ACTIVATION_REPAIR_METHOD,
        "repaired_at": "2026-07-14T00:00:01+00:00",
        "model_revision": request["model_revision"],
        "input_token_ids_sha256": request["input_token_ids_sha256"],
        "rendered_input_sha256": request["rendered_input_sha256"],
        "previous_checksum_sha256": request["expected_checksum_sha256"],
        "replaced_checkpoint": "last_input_token",
        "generated_outputs_preserved": True,
        "generated_checkpoint_tensors_preserved": True,
    }


def _repaired_artifact_bytes(
    request,
    repair_metadata,
    *,
    last_input_value=1.0,
    visible_value=None,
):
    if visible_value is None:
        visible_value = float(request["step_index"] + 10)
    positions = {
        "last_input_token": torch.full(
            (64, 4),
            last_input_value,
            dtype=torch.bfloat16,
        ),
        "last_visible_answer_token": torch.full(
            (64, 4),
            visible_value,
            dtype=torch.bfloat16,
        ),
    }
    buffer = io.BytesIO()
    torch.save(
        {
            "activations": positions[request["primary_checkpoint"]],
            "layers": request["layers"],
            "token_position": request["token_position"],
            "token_id": request["token_id"],
            "artifact_format_version": request["artifact_format_version"],
            "primary_checkpoint": request["primary_checkpoint"],
            "position_activations": positions,
            "checkpoint_positions": request["checkpoint_positions"],
            "checkpoint_forward_scopes": request[
                "expected_checkpoint_forward_scopes"
            ],
            "activation_repair_metadata": repair_metadata,
        },
        buffer,
    )
    return buffer.getvalue()


def _repair_result(item, new_bytes=None):
    request = item["request"]
    repair_metadata = _repair_metadata(request)
    if new_bytes is None:
        new_bytes = _repaired_artifact_bytes(request, repair_metadata)
    new_checksum = hashlib.sha256(new_bytes).hexdigest()
    activation = {
        "storage_status": "materialized",
        "storage_path": request["storage_path"],
        "checksum_sha256": new_checksum,
        "shape": request["checkpoint_shapes"][request["primary_checkpoint"]],
        "dtype": "torch.bfloat16",
        "layers": request["layers"],
        "token_position": request["token_position"],
        "token_id": request["token_id"],
        "artifact_format_version": request["artifact_format_version"],
        "primary_checkpoint": request["primary_checkpoint"],
        "checkpoint_positions": request["checkpoint_positions"],
        "checkpoint_forward_scopes": request[
            "expected_checkpoint_forward_scopes"
        ],
        "checkpoint_shapes": request["checkpoint_shapes"],
        "sequence_length": request["sequence_length"],
    }
    usage = build_token_usage(
        input_token_count=len(request["input_token_ids"]),
        generated_token_count=0,
        thinking_token_count=0,
        final_output_token_count=0,
    )
    cost = build_gpu_cost_record(
        trajectory_id=request["trajectory_id"],
        step_index=request["step_index"],
        agent_id=request["agent_id"],
        thinking_mode=request["thinking_mode"],
        started_at="2026-07-14T00:00:00+00:00",
        finished_at="2026-07-14T00:00:01+00:00",
        elapsed_seconds=1.0,
        modal_input_id="fc-repair-test",
        modal_task_id="ta-repair-test",
        token_usage=usage,
        runtime_metadata={"operation": ACTIVATION_REPAIR_METHOD},
    )
    return {
        "schema_version": ACTIVATION_REPAIR_RESULT_SCHEMA,
        "status": "repaired",
        "trajectory_id": request["trajectory_id"],
        "step_index": request["step_index"],
        "agent_id": request["agent_id"],
        "thinking_mode": request["thinking_mode"],
        "new_checksum_sha256": new_checksum,
        "backup_storage_path": activation_repair_backup_path(
            request["storage_path"]
        ),
        "activation_metadata": activation,
        "repair_metadata": repair_metadata,
        "cost_metadata": cost,
        "artifact_bytes": new_bytes,
    }


def test_plan_selects_only_the_known_clean_injected_smoke_pair(tmp_path):
    output_root = _write_live_pair(tmp_path)
    plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    selected = select_activation_repairs(plan, scope="smoke_pair")
    summary = summarize_activation_repair_plan(
        plan,
        selected,
        scope="smoke_pair",
    )

    assert len(plan) == 6
    assert len(selected) == 2
    assert {item["treatment"] for item in selected} == {"clean", "injected"}
    assert all(item["agent_id"] == "planner_1" for item in selected)
    assert summary["model_generation_required"] is False
    assert summary["existing_generated_outputs_preserved"] is True


def test_repair_request_rejects_path_or_token_tampering(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]

    wrong_path = copy.deepcopy(item["request"])
    wrong_path["storage_path"] = "activations/somewhere-else.pt"
    with pytest.raises(RequestValidationError, match="canonical path"):
        validate_activation_repair_request(wrong_path)

    wrong_tokens = copy.deepcopy(item["request"])
    wrong_tokens["input_token_ids"][-1] += 1
    with pytest.raises(RequestValidationError, match="sha256"):
        validate_activation_repair_request(wrong_tokens)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda request: request["layers"].__setitem__(0, False),
            "requires all 64 Qwen layers",
        ),
        (
            lambda request: request.__setitem__("token_id", True),
            "primary token_id is inconsistent",
        ),
        (
            lambda request: request.__setitem__(
                "token_position", "last_input_token"
            ),
            "token_position must be",
        ),
        (
            lambda request: request["checkpoint_shapes"][
                "last_input_token"
            ].__setitem__(1, True),
            "invalid shape",
        ),
        (
            lambda request: request.__setitem__("input_token_ids", None),
            "token IDs must be",
        ),
        (
            lambda request: request["checkpoint_positions"].__setitem__(0, 7),
            "checkpoint_positions must contain objects",
        ),
    ],
)
def test_repair_request_rejects_bool_and_position_tampering(
    tmp_path,
    mutate,
    message,
):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    tampered = copy.deepcopy(item["request"])
    mutate(tampered)

    with pytest.raises(RequestValidationError, match=message):
        validate_activation_repair_request(tampered)


def test_local_source_preflight_checks_the_saved_checksum(tmp_path):
    output_root = _write_live_pair(tmp_path)
    selected = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )

    result = validate_local_activation_repair_sources(
        selected,
        artifact_root=tmp_path,
    )
    assert result["status"] == "passed"
    assert result["verified_source_count"] == 2

    stale_path = tmp_path / selected[0]["request"]["storage_path"]
    stale_path.write_bytes(b"stale")
    with pytest.raises(ValueError, match="checksum is stale"):
        validate_local_activation_repair_sources(
            selected,
            artifact_root=tmp_path,
        )


def test_apply_repair_updates_artifact_checkpoint_and_live_record(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    payload = _repair_result(item)
    summary = apply_activation_repair_result(
        item,
        payload,
        artifact_root=tmp_path,
    )

    repaired_path = tmp_path / item["request"]["storage_path"]
    backup_path = tmp_path / payload["backup_storage_path"]
    assert repaired_path.read_bytes() == payload["artifact_bytes"]
    assert hashlib.sha256(backup_path.read_bytes()).hexdigest() == item[
        "request"
    ]["expected_checksum_sha256"]
    assert summary["new_checksum_sha256"] == payload["new_checksum_sha256"]

    saved = json.loads(Path(item["checkpoint_path"]).read_text())
    assert saved["activation_metadata"]["checkpoint_forward_scopes"][
        "last_input_token"
    ] == "prompt_only"
    assert saved["activation_repair_metadata"]["generated_outputs_preserved"] is True

    live = json.loads(Path(item["live_path"]).read_text())
    turn = next(
        event
        for event in live["trajectory_trace"]["full_events"]
        if event.get("type") == "agent_turn"
        and event["step_index"] == item["step_index"]
    )
    assert turn["activation_metadata"]["layer_metadata"][
        "checksum_sha256"
    ] == payload["new_checksum_sha256"]
    assert turn["activation_metadata"]["checkpoint_forward_scopes"][
        "last_input_token"
    ] == "prompt_only"


def test_repair_result_rejects_corrupted_returned_bytes(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    payload = _repair_result(item)
    payload["artifact_bytes"] = b"corrupted"

    with pytest.raises(ValueError, match="checksum"):
        apply_activation_repair_result(item, payload, artifact_root=tmp_path)


def test_apply_repair_rejects_a_changed_generated_checkpoint(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    request = item["request"]
    changed_bytes = _repaired_artifact_bytes(
        request,
        _repair_metadata(request),
        visible_value=999.0,
    )
    payload = _repair_result(item, new_bytes=changed_bytes)

    with pytest.raises(ValueError, match="changed generated checkpoint"):
        apply_activation_repair_result(item, payload, artifact_root=tmp_path)


def test_apply_repair_rejects_changed_artifact_token_metadata(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    request = item["request"]
    artifact = torch.load(
        io.BytesIO(
            _repaired_artifact_bytes(request, _repair_metadata(request))
        ),
        map_location="cpu",
        weights_only=True,
    )
    artifact["token_position"] = "last_input_token"
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    payload = _repair_result(item, new_bytes=buffer.getvalue())

    with pytest.raises(ValueError, match="artifact metadata changed"):
        apply_activation_repair_result(item, payload, artifact_root=tmp_path)


def test_apply_repair_can_resume_after_artifact_write(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    payload = _repair_result(item)
    artifact_path = tmp_path / item["request"]["storage_path"]
    backup_path = tmp_path / payload["backup_storage_path"]
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(artifact_path.read_bytes())
    artifact_path.write_bytes(payload["artifact_bytes"])

    summary = apply_activation_repair_result(
        item,
        payload,
        artifact_root=tmp_path,
    )

    assert summary["new_checksum_sha256"] == payload["new_checksum_sha256"]
    saved = json.loads(Path(item["checkpoint_path"]).read_text())
    assert saved["activation_repair_metadata"]["repair_method"] == (
        ACTIVATION_REPAIR_METHOD
    )


def test_reconcile_completes_checkpoint_written_live_missing_transaction(tmp_path):
    output_root = _write_live_pair(tmp_path)
    item = select_activation_repairs(
        build_activation_repair_plan(
            output_root / "live",
            output_root / "checkpoints",
        ),
        scope="smoke_pair",
    )[0]
    original_live = Path(item["live_path"]).read_bytes()
    payload = _repair_result(item)
    apply_activation_repair_result(item, payload, artifact_root=tmp_path)
    Path(item["live_path"]).write_bytes(original_live)

    partial_plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    partial = next(
        candidate
        for candidate in partial_plan
        if candidate["trajectory_id"] == item["trajectory_id"]
        and candidate["step_index"] == item["step_index"]
    )
    assert partial["status"] == "metadata_sync_required"

    summary = reconcile_activation_repair_metadata(
        partial,
        artifact_root=tmp_path,
    )

    assert summary["status"] == "metadata_reconciled"
    refreshed = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    completed = next(
        candidate
        for candidate in refreshed
        if candidate["trajectory_id"] == item["trajectory_id"]
        and candidate["step_index"] == item["step_index"]
    )
    assert completed["status"] == "complete"
    live = json.loads(Path(item["live_path"]).read_text())
    event = next(
        candidate
        for candidate in live["trajectory_trace"]["full_events"]
        if candidate.get("type") == "agent_turn"
        and candidate["step_index"] == item["step_index"]
    )
    assert event["activation_metadata"]["checkpoint_forward_scopes"][
        "last_input_token"
    ] == "prompt_only"
    assert event["output"]["generated_token_ids"] == [100 + item["step_index"]]


def test_local_validation_can_verify_pending_and_complete_plan_items(tmp_path):
    output_root = _write_live_pair(tmp_path)
    plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    item = select_activation_repairs(plan, scope="smoke_pair")[0]
    apply_activation_repair_result(
        item,
        _repair_result(item),
        artifact_root=tmp_path,
    )
    mixed_plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )

    result = validate_local_activation_repair_sources(
        mixed_plan,
        artifact_root=tmp_path,
    )

    assert result["verified_source_count"] == len(mixed_plan) == 6


def test_smoke_verification_requires_exact_matched_planner_activations(tmp_path):
    output_root = _write_live_pair(tmp_path)
    plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    for item in select_activation_repairs(plan, scope="smoke_pair"):
        apply_activation_repair_result(
            item,
            _repair_result(item),
            artifact_root=tmp_path,
        )

    repaired_plan = build_activation_repair_plan(
        output_root / "live",
        output_root / "checkpoints",
    )
    verification = verify_activation_repair_smoke_pair(
        repaired_plan,
        artifact_root=tmp_path,
    )

    assert verification["status"] == "passed"
    assert verification["prompt_hash_match"] is True
    assert verification["input_token_ids_match"] is True
    assert verification["all_layers_exactly_equal"] is True
    assert verification["maximum_absolute_difference"] == 0.0


def test_remote_repair_contract_import_does_not_require_jsonschema():
    script = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "jsonschema" or name.startswith("jsonschema."):
        raise RuntimeError("jsonschema is unavailable in the Modal image")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import src.scenario1.activation_repair
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
