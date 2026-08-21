"""Plan and apply generation-free repairs to saved activation artifacts.

The first live batch extracted every checkpoint during a generated-prefix
forward pass. This module builds a strict repair request from the saved model
turn, then updates the local artifact and both JSON views after Modal returns a
prompt-only ``last_input_token`` tensor. Model responses are never regenerated.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

from src.infrastructure.modal_billing import ANALYSIS_TIERS
from src.infrastructure.modal_costs import validate_gpu_cost_record
from src.infrastructure.qwen_modal import (
    ACTIVATION_ARTIFACT_FORMAT,
    ACTIVATION_TOKEN_POSITION,
    MODEL_LAYER_COUNT,
    MODEL_REVISION,
    RequestValidationError,
    validate_generation_result,
)


ACTIVATION_REPAIR_REQUEST_SCHEMA = "spec_gap.activation_repair_request.v1"
ACTIVATION_REPAIR_RESULT_SCHEMA = "spec_gap.activation_repair_result.v1"
ACTIVATION_REPAIR_VERIFICATION_SCHEMA = (
    "spec_gap.activation_repair_verification.v1"
)
ACTIVATION_REPAIR_METHOD = "prompt_only_last_input_v1"
ACTIVATION_REPAIR_SMOKE_PAIR = "climate_coastal_flood_01__2hop"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHECKPOINT_NAMES = {
    "last_input_token",
    "last_reasoning_token",
    "last_visible_answer_token",
}


def checkpoint_forward_scopes(
    checkpoints: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Return the controlled forward-pass scope for each checkpoint."""

    names: list[str] = []
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise RequestValidationError("checkpoints must contain objects")
        name = checkpoint.get("name")
        if not isinstance(name, str) or not name:
            raise RequestValidationError(
                "checkpoint names must be non-empty strings"
            )
        names.append(name)
    if not names:
        raise RequestValidationError("checkpoint names must be non-empty strings")
    if len(names) != len(set(names)) or "last_input_token" not in names:
        raise RequestValidationError(
            "checkpoints must be unique and include last_input_token"
        )
    return {
        name: "prompt_only" if name == "last_input_token" else "generated_prefix"
        for name in names
    }


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """Hash a token sequence with an unambiguous JSON encoding."""

    try:
        normalized = list(token_ids)
    except TypeError as error:
        raise RequestValidationError(
            "token IDs must be a non-empty list of non-negative integers"
        ) from error
    if not normalized or any(
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id < 0
        for token_id in normalized
    ):
        raise RequestValidationError(
            "token IDs must be a non-empty list of non-negative integers"
        )
    encoded = json.dumps(normalized, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_activation_repair_request(
    generation_result: dict[str, Any],
) -> dict[str, Any]:
    """Build the minimal remote request from one validated saved model turn."""

    result = validate_generation_result(generation_result)
    activation = result.get("activation_metadata")
    if not isinstance(activation, dict):
        raise RequestValidationError(
            "activation repair requires materialized activation metadata"
        )
    checkpoints = copy.deepcopy(activation["checkpoint_positions"])
    expected_scopes = checkpoint_forward_scopes(checkpoints)
    request = {
        "schema_version": ACTIVATION_REPAIR_REQUEST_SCHEMA,
        "repair_method": ACTIVATION_REPAIR_METHOD,
        "trajectory_id": result["trajectory_id"],
        "step_index": result["step_index"],
        "agent_id": result["agent_id"],
        "thinking_mode": result["thinking_mode"],
        "analysis_tier": result["analysis_tier"],
        "model_revision": result["model_revision"],
        "input_token_ids": list(result["input_token_ids"]),
        "input_token_ids_sha256": token_ids_sha256(result["input_token_ids"]),
        "rendered_input_sha256": result["rendered_input_sha256"],
        "storage_path": activation["storage_path"],
        "expected_checksum_sha256": activation["checksum_sha256"],
        "artifact_format_version": activation["artifact_format_version"],
        "layers": list(activation["layers"]),
        "token_position": activation["token_position"],
        "token_id": activation["token_id"],
        "primary_checkpoint": activation["primary_checkpoint"],
        "checkpoint_positions": checkpoints,
        "checkpoint_shapes": copy.deepcopy(activation["checkpoint_shapes"]),
        "expected_checkpoint_forward_scopes": expected_scopes,
        "existing_checkpoint_forward_scopes": copy.deepcopy(
            activation.get("checkpoint_forward_scopes")
        ),
        "sequence_length": activation["sequence_length"],
    }
    return validate_activation_repair_request(request)


def validate_activation_repair_request(payload: Any) -> dict[str, Any]:
    """Reject malformed repair work before a remote GPU method is called."""

    if not isinstance(payload, dict):
        raise RequestValidationError("activation repair request must be an object")
    request = copy.deepcopy(payload)
    if request.get("schema_version") != ACTIVATION_REPAIR_REQUEST_SCHEMA:
        raise RequestValidationError(
            f"schema_version must be {ACTIVATION_REPAIR_REQUEST_SCHEMA!r}"
        )
    if request.get("repair_method") != ACTIVATION_REPAIR_METHOD:
        raise RequestValidationError(
            f"repair_method must be {ACTIVATION_REPAIR_METHOD!r}"
        )
    for field in ("trajectory_id", "agent_id"):
        value = request.get(field)
        if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
            raise RequestValidationError(f"{field} is not a safe identifier")
    mode = request.get("thinking_mode")
    if mode not in {"off", "on"}:
        raise RequestValidationError("thinking_mode must be 'off' or 'on'")
    analysis_tier = request.get("analysis_tier")
    if analysis_tier is not None and analysis_tier not in ANALYSIS_TIERS:
        raise RequestValidationError(
            f"analysis_tier must be one of {sorted(ANALYSIS_TIERS)}"
        )
    step_index = request.get("step_index")
    if (
        not isinstance(step_index, int)
        or isinstance(step_index, bool)
        or step_index < 0
    ):
        raise RequestValidationError("step_index must be a non-negative integer")
    if request.get("model_revision") != MODEL_REVISION:
        raise RequestValidationError(
            f"model_revision must be the pinned revision {MODEL_REVISION!r}"
        )

    input_token_ids = request.get("input_token_ids")
    if not isinstance(input_token_ids, list):
        raise RequestValidationError(
            "token IDs must be a non-empty list of non-negative integers"
        )
    expected_token_hash = token_ids_sha256(input_token_ids)
    if request.get("input_token_ids_sha256") != expected_token_hash:
        raise RequestValidationError("input_token_ids_sha256 does not match")
    for field in ("rendered_input_sha256", "expected_checksum_sha256"):
        if not isinstance(request.get(field), str) or not _SHA256.fullmatch(
            request[field]
        ):
            raise RequestValidationError(f"{field} must be a SHA-256 digest")

    path_parts = ["activations"]
    if analysis_tier is not None:
        path_parts.append(analysis_tier)
    expected_path = PurePosixPath(
        *path_parts,
        request["trajectory_id"],
        mode,
        f"step_{step_index:03d}.pt",
    ).as_posix()
    if request.get("storage_path") != expected_path:
        raise RequestValidationError(
            f"storage_path must be the canonical path {expected_path!r}"
        )
    if request.get("artifact_format_version") != ACTIVATION_ARTIFACT_FORMAT:
        raise RequestValidationError(
            "artifact_format_version is not the current multi-position format"
        )
    layers = request.get("layers")
    if (
        not isinstance(layers, list)
        or any(
            not isinstance(layer, int) or isinstance(layer, bool)
            for layer in layers
        )
        or layers != list(range(MODEL_LAYER_COUNT))
    ):
        raise RequestValidationError(
            f"repair requires all {MODEL_LAYER_COUNT} Qwen layers"
        )
    if request.get("token_position") != ACTIVATION_TOKEN_POSITION:
        raise RequestValidationError(
            f"token_position must be {ACTIVATION_TOKEN_POSITION!r}"
        )
    checkpoints = request.get("checkpoint_positions")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise RequestValidationError("checkpoint_positions must be non-empty")
    if any(not isinstance(checkpoint, dict) for checkpoint in checkpoints):
        raise RequestValidationError(
            "checkpoint_positions must contain objects"
        )
    expected_scopes = checkpoint_forward_scopes(checkpoints)
    if not set(expected_scopes).issubset(_CHECKPOINT_NAMES):
        raise RequestValidationError("checkpoint_positions contain an unknown name")
    sequence_indices = []
    for checkpoint in checkpoints:
        sequence_index = checkpoint.get("sequence_index")
        generated_offset = checkpoint.get("generated_token_offset")
        token_id = checkpoint.get("token_id")
        if (
            not isinstance(sequence_index, int)
            or isinstance(sequence_index, bool)
            or sequence_index < 0
            or not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
        ):
            raise RequestValidationError("checkpoint token metadata is invalid")
        if checkpoint["name"] == "last_input_token":
            if generated_offset is not None:
                raise RequestValidationError(
                    "last_input_token cannot have a generated offset"
                )
        elif (
            not isinstance(generated_offset, int)
            or isinstance(generated_offset, bool)
            or generated_offset < 0
            or sequence_index != len(input_token_ids) + generated_offset
        ):
            raise RequestValidationError(
                "generated checkpoint offset is inconsistent"
            )
        sequence_indices.append(sequence_index)
    if len(sequence_indices) != len(set(sequence_indices)):
        raise RequestValidationError("checkpoint sequence indices must be unique")
    if request.get("expected_checkpoint_forward_scopes") != expected_scopes:
        raise RequestValidationError(
            "expected_checkpoint_forward_scopes are inconsistent"
        )
    existing_scopes = request.get("existing_checkpoint_forward_scopes")
    if existing_scopes not in (None, {}, expected_scopes):
        raise RequestValidationError(
            "existing checkpoint scopes are partial or inconsistent"
        )

    input_checkpoint = next(
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.get("name") == "last_input_token"
    )
    if (
        input_checkpoint.get("sequence_index") != len(input_token_ids) - 1
        or input_checkpoint.get("generated_token_offset") is not None
        or input_checkpoint.get("token_id") != input_token_ids[-1]
    ):
        raise RequestValidationError(
            "last_input_token does not identify the final prompt token"
        )
    names = [checkpoint["name"] for checkpoint in checkpoints]
    primary = request.get("primary_checkpoint")
    if primary not in names or primary == "last_input_token":
        raise RequestValidationError(
            "primary_checkpoint must identify a generated-token checkpoint"
        )
    by_name = {checkpoint["name"]: checkpoint for checkpoint in checkpoints}
    primary_token_id = request.get("token_id")
    if (
        not isinstance(primary_token_id, int)
        or isinstance(primary_token_id, bool)
        or primary_token_id < 0
        or primary_token_id != by_name[primary].get("token_id")
    ):
        raise RequestValidationError("primary token_id is inconsistent")
    shapes = request.get("checkpoint_shapes")
    if not isinstance(shapes, dict) or set(shapes) != set(names):
        raise RequestValidationError(
            "checkpoint_shapes must cover every checkpoint"
        )
    for name, shape in shapes.items():
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != MODEL_LAYER_COUNT
            or any(
                not isinstance(size, int)
                or isinstance(size, bool)
                or size < 1
                for size in shape
            )
        ):
            raise RequestValidationError(f"checkpoint {name!r} has an invalid shape")
    sequence_length = request.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length
        != max(checkpoint["sequence_index"] for checkpoint in checkpoints) + 1
    ):
        raise RequestValidationError("sequence_length is inconsistent")
    return request


def activation_repair_backup_path(storage_path: str) -> str:
    """Return the immutable Volume/local backup path for an original artifact."""

    path = PurePosixPath(storage_path)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[0] != "activations"
    ):
        raise RequestValidationError("storage_path is not a safe activation path")
    return PurePosixPath(
        "activation_backups", ACTIVATION_REPAIR_METHOD, *path.parts
    ).as_posix()


def build_activation_repair_plan(
    trajectory_root: str | Path,
    checkpoint_root: str | Path,
) -> list[dict[str, Any]]:
    """Cross-check live trajectories and turn checkpoints, then plan repairs."""

    live_root = Path(trajectory_root)
    saved_root = Path(checkpoint_root)
    from src.scenario1.validator import validate_payload

    live_paths = sorted(live_root.rglob("*.json"))
    if not live_paths:
        raise ValueError(f"No live trajectory JSON files found under {live_root}.")

    plan = []
    seen: set[tuple[str, str, int]] = set()
    for live_path in live_paths:
        record = _load_json_object(live_path, "live trajectory")
        errors = validate_payload(record)
        if errors:
            raise ValueError(
                f"Invalid live trajectory {live_path}: " + "; ".join(errors)
            )
        trajectory_id = str(record["trajectory_id"])
        mode = str(record["model"]["thinking_mode"])
        turns = [
            event
            for event in record["trajectory_trace"]["full_events"]
            if event.get("type") == "agent_turn"
        ]
        turn_tiers = {
            event.get("model_execution_metadata", {}).get("analysis_tier")
            for event in turns
        }
        if len(turn_tiers) != 1:
            raise ValueError(
                f"Live trajectory {live_path} mixes analysis tiers."
            )
        analysis_tier = next(iter(turn_tiers))
        checkpoint_base = saved_root
        if analysis_tier is not None:
            checkpoint_base /= str(analysis_tier)
        for event in turns:
            step_index = int(event["step_index"])
            key = (
                str(analysis_tier or "unclassified"),
                trajectory_id,
                step_index,
            )
            if key in seen:
                raise ValueError(f"Duplicate repair-plan key {key!r}.")
            seen.add(key)
            checkpoint_path = (
                checkpoint_base
                / trajectory_id
                / mode
                / f"step_{step_index:03d}.json"
            )
            saved_result = _load_json_object(checkpoint_path, "model-turn checkpoint")
            normalized_result = validate_generation_result(saved_result)
            request = build_activation_repair_request(normalized_result)
            event_activation = event["activation_metadata"]
            expected_scopes = request["expected_checkpoint_forward_scopes"]
            raw_scopes = request["existing_checkpoint_forward_scopes"]
            event_scopes = event_activation.get("checkpoint_forward_scopes")
            metadata_sync_candidate = (
                raw_scopes == expected_scopes and event_scopes in (None, {})
            )
            _validate_turn_views(
                record,
                event,
                normalized_result,
                allow_original_activation_checksum=metadata_sync_candidate,
            )
            if raw_scopes == expected_scopes and event_scopes == expected_scopes:
                status = "complete"
            elif raw_scopes in (None, {}) and event_scopes in (None, {}):
                status = "pending"
            elif raw_scopes == expected_scopes and event_scopes in (None, {}):
                status = "metadata_sync_required"
            else:
                raise ValueError(
                    f"Partial or inconsistent activation repair state for {key!r}."
                )
            plan.append({
                "trajectory_id": trajectory_id,
                "step_index": step_index,
                "agent_id": event["agent_id"],
                "thinking_mode": mode,
                "analysis_tier": analysis_tier or "unclassified",
                "matched_pair_id": record["matched_pair_id"],
                "treatment": record["treatment"],
                "status": status,
                "live_path": live_path,
                "checkpoint_path": checkpoint_path,
                "request": request,
            })
    return sorted(
        plan,
        key=lambda item: (
            item["analysis_tier"],
            item["thinking_mode"],
            item["matched_pair_id"],
            item["treatment"],
            item["step_index"],
        ),
    )


def select_activation_repairs(
    plan: Iterable[dict[str, Any]],
    *,
    scope: str,
    max_repairs: int = 0,
) -> list[dict[str, Any]]:
    """Select pending work for the fixed smoke pair or the complete batch."""

    if scope not in {"smoke_pair", "all"}:
        raise ValueError("scope must be 'smoke_pair' or 'all'")
    if (
        not isinstance(max_repairs, int)
        or isinstance(max_repairs, bool)
        or max_repairs < 0
    ):
        raise ValueError("max_repairs must be zero or greater")
    pending = [item for item in plan if item["status"] == "pending"]
    if scope == "smoke_pair":
        selected = [
            item
            for item in pending
            if item["matched_pair_id"] == ACTIVATION_REPAIR_SMOKE_PAIR
            and item["thinking_mode"] == "off"
            and item["agent_id"] == "planner_1"
        ]
        if selected and len(selected) != 2:
            raise ValueError(
                "the smoke scope must select the clean/injected planner pair"
            )
    else:
        selected = pending
    return selected[:max_repairs] if max_repairs else selected


def summarize_activation_repair_plan(
    plan: Iterable[dict[str, Any]],
    selected: Iterable[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Return a compact, JSON-safe validation preview."""

    items = list(plan)
    chosen = list(selected)
    statuses = {
        status: sum(item["status"] == status for item in items)
        for status in ("pending", "complete", "metadata_sync_required")
    }
    return {
        "repair_method": ACTIVATION_REPAIR_METHOD,
        "scope": scope,
        "total_model_turns": len(items),
        "status_counts": statuses,
        "selected_repairs": len(chosen),
        "selected_ids": [
            f"{item['trajectory_id']}:step_{item['step_index']:03d}"
            for item in chosen
        ],
        "model_generation_required": False,
        "existing_generated_outputs_preserved": True,
    }


def validate_local_activation_repair_sources(
    selected: Iterable[dict[str, Any]],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Verify local source artifacts before any selected remote call can start."""

    root = Path(artifact_root).resolve()
    verified = []
    for item in selected:
        request = item["request"]
        path = (root / request["storage_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("activation repair source path escapes artifact_root")
        if not path.is_file():
            raise FileNotFoundError(
                f"local activation repair source is missing: {path}"
            )
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != request["expected_checksum_sha256"]:
            raise ValueError(
                f"local activation repair source checksum is stale: {path}"
            )
        verified.append(path.as_posix())
    return {
        "status": "passed",
        "verified_source_count": len(verified),
        "verified_source_paths": verified,
    }


def verify_activation_repair_smoke_pair(
    plan: Iterable[dict[str, Any]],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Verify that the repaired matched planner inputs are bitwise identical."""

    candidates = [
        item
        for item in plan
        if item["matched_pair_id"] == ACTIVATION_REPAIR_SMOKE_PAIR
        and item["thinking_mode"] == "off"
        and item["agent_id"] == "planner_1"
    ]
    if len(candidates) != 2 or {
        item["treatment"] for item in candidates
    } != {"clean", "injected"}:
        raise ValueError(
            "smoke verification requires one clean and one injected planner turn"
        )
    if any(item["status"] != "complete" for item in candidates):
        raise ValueError("smoke verification requires both repairs to be complete")

    by_treatment = {item["treatment"]: item for item in candidates}
    clean = by_treatment["clean"]
    injected = by_treatment["injected"]
    clean_request = clean["request"]
    injected_request = injected["request"]
    prompt_hash_match = (
        clean_request["rendered_input_sha256"]
        == injected_request["rendered_input_sha256"]
    )
    input_token_hash_match = (
        clean_request["input_token_ids_sha256"]
        == injected_request["input_token_ids_sha256"]
    )
    input_token_ids_match = (
        clean_request["input_token_ids"] == injected_request["input_token_ids"]
    )

    from src.extraction.saved_activations import load_activation_checkpoint

    root = Path(artifact_root).resolve()
    loaded = {}
    for treatment, item in by_treatment.items():
        request = item["request"]
        path = (root / request["storage_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("smoke activation path escapes artifact_root")
        loaded[treatment] = load_activation_checkpoint(
            path,
            checkpoint="last_input_token",
            expected_checksum=request["expected_checksum_sha256"],
            verify_checksum=True,
        )

    clean_loaded = loaded["clean"]
    injected_loaded = loaded["injected"]
    if clean_loaded.layers != injected_loaded.layers:
        raise ValueError("smoke pair activation layers do not match")
    difference = (
        injected_loaded.activations.float()
        - clean_loaded.activations.float()
    )
    exact_by_layer = {
        str(layer): bool(
            clean_loaded.activations[offset].equal(
                injected_loaded.activations[offset]
            )
        )
        for offset, layer in enumerate(clean_loaded.layers)
    }
    exact_match = all(exact_by_layer.values())
    passed = (
        prompt_hash_match
        and input_token_hash_match
        and input_token_ids_match
        and exact_match
    )
    return {
        "schema_version": ACTIVATION_REPAIR_VERIFICATION_SCHEMA,
        "repair_method": ACTIVATION_REPAIR_METHOD,
        "matched_pair_id": ACTIVATION_REPAIR_SMOKE_PAIR,
        "thinking_mode": "off",
        "agent_id": "planner_1",
        "status": "passed" if passed else "failed",
        "prompt_hash_match": prompt_hash_match,
        "input_token_hash_match": input_token_hash_match,
        "input_token_ids_match": input_token_ids_match,
        "all_layers_exactly_equal": exact_match,
        "exact_match_by_layer": exact_by_layer,
        "maximum_absolute_difference": float(difference.abs().max().item()),
        "clean_trajectory_id": clean["trajectory_id"],
        "injected_trajectory_id": injected["trajectory_id"],
    }


def validate_activation_repair_result(
    payload: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Validate a repaired artifact and its returned provenance before writing."""

    normalized_request = validate_activation_repair_request(request)
    if not isinstance(payload, dict):
        raise ValueError("activation repair result must be an object")
    result = copy.deepcopy(payload)
    if result.get("schema_version") != ACTIVATION_REPAIR_RESULT_SCHEMA:
        raise ValueError(
            f"repair result schema must be {ACTIVATION_REPAIR_RESULT_SCHEMA!r}"
        )
    if result.get("status") not in {"repaired", "already_repaired"}:
        raise ValueError("repair result status is invalid")
    for field in ("trajectory_id", "step_index", "agent_id", "thinking_mode"):
        if result.get(field) != normalized_request[field]:
            raise ValueError(f"repair result {field} does not match request")
    artifact_bytes = result.get("artifact_bytes")
    if not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        raise ValueError("repair result must include non-empty artifact_bytes")
    new_checksum = hashlib.sha256(artifact_bytes).hexdigest()
    if result.get("new_checksum_sha256") != new_checksum:
        raise ValueError("repaired artifact checksum does not match returned bytes")
    if not _SHA256.fullmatch(new_checksum):
        raise ValueError("new checksum is invalid")
    expected_backup = activation_repair_backup_path(
        normalized_request["storage_path"]
    )
    if result.get("backup_storage_path") != expected_backup:
        raise ValueError("repair backup path is inconsistent")

    activation = result.get("activation_metadata")
    if not isinstance(activation, dict):
        raise ValueError("repair result activation_metadata must be an object")
    exact_fields = {
        "storage_status": "materialized",
        "storage_path": normalized_request["storage_path"],
        "checksum_sha256": new_checksum,
        "layers": normalized_request["layers"],
        "token_position": normalized_request["token_position"],
        "token_id": normalized_request["token_id"],
        "artifact_format_version": normalized_request["artifact_format_version"],
        "primary_checkpoint": normalized_request["primary_checkpoint"],
        "checkpoint_positions": normalized_request["checkpoint_positions"],
        "checkpoint_forward_scopes": normalized_request[
            "expected_checkpoint_forward_scopes"
        ],
        "checkpoint_shapes": normalized_request["checkpoint_shapes"],
        "sequence_length": normalized_request["sequence_length"],
    }
    for field, expected in exact_fields.items():
        if activation.get(field) != expected:
            raise ValueError(f"repaired activation metadata {field} is inconsistent")
    if activation.get("shape") != normalized_request["checkpoint_shapes"][
        normalized_request["primary_checkpoint"]
    ]:
        raise ValueError("repaired primary activation shape is inconsistent")
    if activation.get("dtype") != "torch.bfloat16":
        raise ValueError("repaired activations must use torch.bfloat16")

    repair_metadata = result.get("repair_metadata")
    if not isinstance(repair_metadata, dict):
        raise ValueError("repair_metadata must be an object")
    if (
        repair_metadata.get("schema_version") != ACTIVATION_REPAIR_RESULT_SCHEMA
        or repair_metadata.get("repair_method") != ACTIVATION_REPAIR_METHOD
        or repair_metadata.get("model_revision") != MODEL_REVISION
        or repair_metadata.get("input_token_ids_sha256")
        != normalized_request["input_token_ids_sha256"]
        or repair_metadata.get("rendered_input_sha256")
        != normalized_request["rendered_input_sha256"]
        or repair_metadata.get("previous_checksum_sha256")
        != normalized_request["expected_checksum_sha256"]
        or repair_metadata.get("replaced_checkpoint") != "last_input_token"
        or repair_metadata.get("generated_outputs_preserved") is not True
        or repair_metadata.get("generated_checkpoint_tensors_preserved") is not True
        or not isinstance(repair_metadata.get("repaired_at"), str)
        or not repair_metadata["repaired_at"]
    ):
        raise ValueError("repair provenance does not match the request")
    cost = validate_gpu_cost_record(result.get("cost_metadata"))
    if (
        cost["trajectory_id"] != normalized_request["trajectory_id"]
        or cost["step_index"] != normalized_request["step_index"]
        or cost["agent_id"] != normalized_request["agent_id"]
        or cost["thinking_mode"] != normalized_request["thinking_mode"]
        or cost["token_usage"]["input_tokens"]
        != len(normalized_request["input_token_ids"])
        or cost["token_usage"]["generated_tokens"] != 0
        or cost.get("runtime_metadata", {}).get("operation")
        != ACTIVATION_REPAIR_METHOD
        or cost.get("runtime_metadata", {}).get("analysis_tier")
        != normalized_request.get("analysis_tier")
    ):
        raise ValueError("repair cost metadata is inconsistent")
    result["cost_metadata"] = cost
    return result


def apply_activation_repair_result(
    item: dict[str, Any],
    payload: dict[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Atomically apply one remote repair to artifact, checkpoint, and live JSON."""

    request = item["request"]
    result = validate_activation_repair_result(payload, request)
    from src.scenario1.validator import validate_payload

    root = Path(artifact_root).resolve()
    artifact_path = (root / request["storage_path"]).resolve()
    backup_path = (root / result["backup_storage_path"]).resolve()
    if not artifact_path.is_relative_to(root) or not backup_path.is_relative_to(root):
        raise ValueError("repair paths escape artifact_root")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"local activation artifact is missing: {artifact_path}")
    old_bytes = artifact_path.read_bytes()
    old_checksum = hashlib.sha256(old_bytes).hexdigest()
    if old_checksum not in {
        request["expected_checksum_sha256"],
        result["new_checksum_sha256"],
    }:
        raise ValueError("local artifact matches neither old nor repaired checksum")
    if old_checksum == request["expected_checksum_sha256"]:
        if backup_path.exists():
            if hashlib.sha256(backup_path.read_bytes()).hexdigest() != old_checksum:
                raise ValueError("existing local activation backup has wrong checksum")
        else:
            _atomic_write_bytes(backup_path, old_bytes)
        original_bytes = old_bytes
    else:
        if (
            not backup_path.is_file()
            or hashlib.sha256(backup_path.read_bytes()).hexdigest()
            != request["expected_checksum_sha256"]
        ):
            raise ValueError("repaired local artifact has no valid original backup")
        if old_bytes != result["artifact_bytes"]:
            raise ValueError("local repaired artifact differs from returned artifact")
        original_bytes = backup_path.read_bytes()
    _validate_repaired_artifact_bytes(
        original_bytes,
        result["artifact_bytes"],
        request=request,
        repair_metadata=result["repair_metadata"],
    )
    _atomic_write_bytes(artifact_path, result["artifact_bytes"])

    checkpoint_path = Path(item["checkpoint_path"])
    saved_result = _load_json_object(checkpoint_path, "model-turn checkpoint")
    saved_result["activation_metadata"] = copy.deepcopy(
        result["activation_metadata"]
    )
    saved_result["activation_repair_metadata"] = copy.deepcopy(
        result["repair_metadata"]
    )
    saved_result["activation_repair_cost_metadata"] = copy.deepcopy(
        result["cost_metadata"]
    )
    saved_result = validate_generation_result(saved_result)

    live_path = Path(item["live_path"])
    record = _load_json_object(live_path, "live trajectory")
    event = next(
        (
            candidate
            for candidate in record["trajectory_trace"]["full_events"]
            if candidate.get("type") == "agent_turn"
            and candidate.get("step_index") == request["step_index"]
            and candidate.get("agent_id") == request["agent_id"]
        ),
        None,
    )
    if event is None:
        raise ValueError("live trajectory no longer contains the repaired turn")
    _update_live_activation_metadata(
        event,
        activation_metadata=result["activation_metadata"],
        repair_metadata=result["repair_metadata"],
        cost_metadata=result["cost_metadata"],
    )
    _validate_turn_views(record, event, saved_result)
    errors = validate_payload(record)
    if errors:
        raise ValueError(
            "repaired live trajectory failed validation: " + "; ".join(errors)
        )

    _atomic_write_json(checkpoint_path, saved_result)
    _atomic_write_json(live_path, record)
    return {
        "trajectory_id": request["trajectory_id"],
        "step_index": request["step_index"],
        "agent_id": request["agent_id"],
        "thinking_mode": request["thinking_mode"],
        "status": result["status"],
        "previous_checksum_sha256": request["expected_checksum_sha256"],
        "new_checksum_sha256": result["new_checksum_sha256"],
        "artifact_path": artifact_path.as_posix(),
        "backup_path": backup_path.as_posix(),
        "estimated_h200_cost_usd": result["cost_metadata"][
            "estimated_h200_cost_usd"
        ],
    }


def reconcile_activation_repair_metadata(
    item: dict[str, Any],
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Finish a checkpoint-written/live-record-missing local repair transaction.

    This path performs no model call. It validates the repaired artifact, its
    immutable original backup, the per-turn checkpoint, and the repair cost
    record before copying the already-saved metadata into the live trajectory.
    """

    if item.get("status") != "metadata_sync_required":
        raise ValueError("metadata reconciliation requires metadata_sync_required")
    current_request = validate_activation_repair_request(item["request"])
    checkpoint_path = Path(item["checkpoint_path"])
    saved_result = validate_generation_result(
        _load_json_object(checkpoint_path, "model-turn checkpoint")
    )
    activation = saved_result.get("activation_metadata")
    repair_metadata = saved_result.get("activation_repair_metadata")
    cost_metadata = saved_result.get("activation_repair_cost_metadata")
    if not isinstance(activation, dict) or not isinstance(repair_metadata, dict):
        raise ValueError("repaired checkpoint is missing activation provenance")
    previous_checksum = repair_metadata.get("previous_checksum_sha256")
    if not isinstance(previous_checksum, str) or not _SHA256.fullmatch(
        previous_checksum
    ):
        raise ValueError("repaired checkpoint has an invalid previous checksum")

    legacy_request = copy.deepcopy(current_request)
    legacy_request["expected_checksum_sha256"] = previous_checksum
    root = Path(artifact_root).resolve()
    artifact_path = (root / current_request["storage_path"]).resolve()
    backup_storage_path = activation_repair_backup_path(
        current_request["storage_path"]
    )
    backup_path = (root / backup_storage_path).resolve()
    if not artifact_path.is_relative_to(root) or not backup_path.is_relative_to(root):
        raise ValueError("repair reconciliation paths escape artifact_root")
    if not artifact_path.is_file() or not backup_path.is_file():
        raise FileNotFoundError(
            "repair reconciliation requires the repaired artifact and its backup"
        )
    artifact_bytes = artifact_path.read_bytes()
    new_checksum = hashlib.sha256(artifact_bytes).hexdigest()
    if new_checksum != current_request["expected_checksum_sha256"]:
        raise ValueError("repaired artifact checksum does not match the checkpoint")

    persisted_result = validate_activation_repair_result(
        {
            "schema_version": ACTIVATION_REPAIR_RESULT_SCHEMA,
            "status": "already_repaired",
            "trajectory_id": current_request["trajectory_id"],
            "step_index": current_request["step_index"],
            "agent_id": current_request["agent_id"],
            "thinking_mode": current_request["thinking_mode"],
            "new_checksum_sha256": new_checksum,
            "backup_storage_path": backup_storage_path,
            "activation_metadata": activation,
            "repair_metadata": repair_metadata,
            "cost_metadata": cost_metadata,
            "artifact_bytes": artifact_bytes,
        },
        legacy_request,
    )
    backup_bytes = backup_path.read_bytes()
    if hashlib.sha256(backup_bytes).hexdigest() != previous_checksum:
        raise ValueError("original activation backup checksum is inconsistent")
    _validate_repaired_artifact_bytes(
        backup_bytes,
        artifact_bytes,
        request=legacy_request,
        repair_metadata=persisted_result["repair_metadata"],
    )

    live_path = Path(item["live_path"])
    record = _load_json_object(live_path, "live trajectory")
    event = next(
        (
            candidate
            for candidate in record["trajectory_trace"]["full_events"]
            if candidate.get("type") == "agent_turn"
            and candidate.get("step_index") == current_request["step_index"]
            and candidate.get("agent_id") == current_request["agent_id"]
        ),
        None,
    )
    if event is None:
        raise ValueError("live trajectory no longer contains the repaired turn")
    _update_live_activation_metadata(
        event,
        activation_metadata=persisted_result["activation_metadata"],
        repair_metadata=persisted_result["repair_metadata"],
        cost_metadata=persisted_result["cost_metadata"],
    )
    _validate_turn_views(record, event, saved_result)
    from src.scenario1.validator import validate_payload

    errors = validate_payload(record)
    if errors:
        raise ValueError(
            "reconciled live trajectory failed validation: " + "; ".join(errors)
        )
    _atomic_write_json(live_path, record)
    return {
        "trajectory_id": current_request["trajectory_id"],
        "step_index": current_request["step_index"],
        "agent_id": current_request["agent_id"],
        "thinking_mode": current_request["thinking_mode"],
        "status": "metadata_reconciled",
        "checksum_sha256": new_checksum,
        "live_path": live_path.as_posix(),
    }


def _update_live_activation_metadata(
    event: dict[str, Any],
    *,
    activation_metadata: dict[str, Any],
    repair_metadata: dict[str, Any],
    cost_metadata: dict[str, Any],
) -> None:
    """Copy validated repaired activation metadata into one live event."""

    activation = event.get("activation_metadata")
    if not isinstance(activation, dict) or not isinstance(
        activation.get("layer_metadata"), dict
    ):
        raise ValueError("live event is missing activation layer metadata")
    activation.update({
        "requested_layers": copy.deepcopy(activation_metadata["layers"]),
        "layers_extracted": copy.deepcopy(activation_metadata["layers"]),
        "token_position_convention": activation_metadata["token_position"],
        "sequence_length": activation_metadata["sequence_length"],
        "storage_path": activation_metadata["storage_path"],
        "storage_status": activation_metadata["storage_status"],
        "artifact_format_version": activation_metadata[
            "artifact_format_version"
        ],
        "primary_checkpoint": activation_metadata["primary_checkpoint"],
        "checkpoint_positions": copy.deepcopy(
            activation_metadata["checkpoint_positions"]
        ),
        "checkpoint_forward_scopes": copy.deepcopy(
            activation_metadata["checkpoint_forward_scopes"]
        ),
        "checkpoint_shapes": copy.deepcopy(
            activation_metadata["checkpoint_shapes"]
        ),
        "activation_repair_metadata": copy.deepcopy(repair_metadata),
        "activation_repair_cost_metadata": copy.deepcopy(cost_metadata),
    })
    activation["layer_metadata"].update({
        "shape": copy.deepcopy(activation_metadata["shape"]),
        "dtype": activation_metadata["dtype"],
        "checksum_sha256": activation_metadata["checksum_sha256"],
        "token_id": activation_metadata["token_id"],
    })


def _validate_repaired_artifact_bytes(
    old_bytes: bytes,
    new_bytes: bytes,
    *,
    request: dict[str, Any],
    repair_metadata: dict[str, Any],
) -> None:
    """Verify tensor contents locally before replacing the downloaded artifact."""

    import io

    import torch

    def load(content: bytes, description: str) -> dict[str, Any]:
        try:
            artifact = torch.load(
                io.BytesIO(content),
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"cannot load {description} activation artifact") from error
        if not isinstance(artifact, dict):
            raise ValueError(f"{description} activation artifact must be an object")
        return artifact

    old = load(old_bytes, "original")
    new = load(new_bytes, "repaired")
    expected_scopes = request["expected_checkpoint_forward_scopes"]
    for description, artifact in (("original", old), ("repaired", new)):
        if (
            artifact.get("artifact_format_version")
            != request["artifact_format_version"]
            or artifact.get("layers") != request["layers"]
            or artifact.get("token_position") != request["token_position"]
            or artifact.get("token_id") != request["token_id"]
            or artifact.get("primary_checkpoint") != request["primary_checkpoint"]
            or artifact.get("checkpoint_positions")
            != request["checkpoint_positions"]
        ):
            raise ValueError(f"{description} activation artifact metadata changed")
        positions = artifact.get("position_activations")
        if not isinstance(positions, dict) or set(positions) != set(
            expected_scopes
        ):
            raise ValueError(
                f"{description} activation artifact checkpoints are inconsistent"
            )
        for name, tensor in positions.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or list(tensor.shape) != request["checkpoint_shapes"][name]
                or tensor.dtype != torch.bfloat16
                or not bool(torch.isfinite(tensor.float()).all())
            ):
                raise ValueError(
                    f"{description} activation checkpoint {name!r} is invalid"
                )
        primary = positions[request["primary_checkpoint"]]
        if not isinstance(artifact.get("activations"), torch.Tensor) or not torch.equal(
            artifact["activations"], primary
        ):
            raise ValueError(
                f"{description} primary activation tensor is inconsistent"
            )

    if old.get("checkpoint_forward_scopes") not in (None, {}):
        raise ValueError("original activation artifact is not a legacy artifact")
    if new.get("checkpoint_forward_scopes") != expected_scopes:
        raise ValueError("repaired activation artifact has the wrong forward scopes")
    if new.get("activation_repair_metadata") != repair_metadata:
        raise ValueError("repaired activation artifact provenance is inconsistent")
    for name in expected_scopes:
        if name == "last_input_token":
            continue
        if not torch.equal(
            old["position_activations"][name],
            new["position_activations"][name],
        ):
            raise ValueError(
                f"repair changed generated checkpoint {name!r}"
            )
    if not torch.equal(old["activations"], new["activations"]):
        raise ValueError("repair changed the primary activation tensor")


def _validate_turn_views(
    record: dict[str, Any],
    event: dict[str, Any],
    result: dict[str, Any],
    *,
    allow_original_activation_checksum: bool = False,
) -> None:
    expected = {
        "trajectory_id": record["trajectory_id"],
        "step_index": event["step_index"],
        "agent_id": event["agent_id"],
        "thinking_mode": record["model"]["thinking_mode"],
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(f"saved turn {field} does not match live trajectory")
    if event["input"].get("input_token_ids") != result["input_token_ids"]:
        raise ValueError("saved and live input token IDs differ")
    if event["input"].get("rendered_prompt_hash") != result[
        "rendered_input_sha256"
    ]:
        raise ValueError("saved and live rendered prompt hashes differ")
    if event["output"].get("generated_token_ids") != result[
        "generated_token_ids"
    ]:
        raise ValueError("saved and live generated token IDs differ")
    raw_activation = result["activation_metadata"]
    event_activation = event.get("activation_metadata")
    if not isinstance(raw_activation, dict) or not isinstance(event_activation, dict):
        raise ValueError("both turn views require activation metadata")
    comparisons = {
        "storage_status": raw_activation.get("storage_status"),
        "storage_path": raw_activation.get("storage_path"),
        "artifact_format_version": raw_activation.get("artifact_format_version"),
        "primary_checkpoint": raw_activation.get("primary_checkpoint"),
        "checkpoint_positions": raw_activation.get("checkpoint_positions"),
        "checkpoint_shapes": raw_activation.get("checkpoint_shapes"),
        "sequence_length": raw_activation.get("sequence_length"),
    }
    for field, expected_value in comparisons.items():
        if event_activation.get(field) != expected_value:
            raise ValueError(f"saved and live activation {field} differ")
    if event_activation.get("requested_layers") != raw_activation.get("layers"):
        raise ValueError("saved and live requested activation layers differ")
    if event_activation.get("layers_extracted") != raw_activation.get("layers"):
        raise ValueError("saved and live extracted activation layers differ")
    if event_activation.get("token_position_convention") != raw_activation.get(
        "token_position"
    ):
        raise ValueError("saved and live activation token positions differ")
    event_layer = event_activation.get("layer_metadata")
    if not isinstance(event_layer, dict):
        raise ValueError("live event is missing activation layer metadata")
    for field in ("shape", "dtype", "token_id"):
        if event_layer.get(field) != raw_activation.get(field):
            raise ValueError(f"saved and live activation {field} differ")
    event_checksum = event_layer.get("checksum_sha256")
    raw_checksum = raw_activation.get("checksum_sha256")
    if event_checksum != raw_checksum:
        repair_metadata = result.get("activation_repair_metadata")
        original_checksum = (
            repair_metadata.get("previous_checksum_sha256")
            if isinstance(repair_metadata, dict)
            else None
        )
        if not (
            allow_original_activation_checksum
            and event_checksum == original_checksum
            and isinstance(original_checksum, str)
            and _SHA256.fullmatch(original_checksum)
        ):
            raise ValueError("saved and live activation checksums differ")


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {description} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} {path} must be an object")
    return payload


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    content = (json.dumps(payload, indent=2) + "\n").encode()
    _atomic_write_bytes(path, content)
