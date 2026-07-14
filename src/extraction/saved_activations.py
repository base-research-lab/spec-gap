"""Load saved Scenario 1 activation checkpoints for probe analysis.

The live Qwen runner stores one ``.pt`` artifact per model turn. Current
artifacts contain all requested layers at two or three named token positions.
This module turns the trajectory metadata into a flat, filterable index and
loads one named checkpoint without starting a model or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from src.infrastructure.qwen_modal import ACTIVATION_ARTIFACT_FORMAT


ACTIVATION_INDEX_SCHEMA = "spec_gap.activation_index.v2"
CHECKPOINT_NAMES = {
    "last_input_token",
    "last_reasoning_token",
    "last_visible_answer_token",
}
LABEL_TARGETS = {
    "injection_present",
    "unsafe_action_executed",
    "output_adoption",
}


@dataclass(frozen=True)
class LoadedActivationCheckpoint:
    """One saved token position across all extracted model layers."""

    checkpoint: str
    activations: Tensor
    layers: tuple[int, ...]
    artifact_format_version: str | None
    source_path: Path


@dataclass(frozen=True)
class ProbeActivationBatch:
    """Layer matrices, binary labels, and aligned index rows."""

    activations: dict[int, np.ndarray]
    labels: np.ndarray
    metadata: list[dict[str, Any]]


def load_trajectory_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load trajectory JSON objects in deterministic path order."""

    records: list[dict[str, Any]] = []
    for path in sorted((Path(path) for path in paths), key=lambda item: item.as_posix()):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load trajectory JSON {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Trajectory JSON {path} must contain one object.")
        records.append(payload)
    if not records:
        raise ValueError("At least one trajectory JSON file is required.")
    return records


def build_activation_index(
    records: Iterable[dict[str, Any]],
    *,
    artifact_root: str | Path,
    require_local: bool = False,
    verify_checksums: bool = False,
) -> list[dict[str, Any]]:
    """Build one index row per model turn and available checkpoint.

    ``artifact_root`` is the local equivalent of the Modal artifact-volume
    root. For the repository layout, pass the repository root because saved
    paths already begin with ``activations/``.
    """

    root = Path(artifact_root).resolve()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for record in records:
        trajectory_id = _required_string(record, "trajectory_id")
        model = _required_mapping(record, "model", trajectory_id)
        labels = _required_mapping(record, "evaluation_labels", trajectory_id)
        action_channel = _required_mapping(labels, "action_channel", trajectory_id)
        behavioral_channel = _required_mapping(
            labels, "behavioral_channel", trajectory_id
        )
        events = (
            record.get("trajectory_trace", {}).get("full_events", [])
            if isinstance(record.get("trajectory_trace"), dict)
            else []
        )
        turns = [event for event in events if event.get("type") == "agent_turn"]
        if not turns:
            raise ValueError(f"Trajectory {trajectory_id!r} has no agent turns.")

        for turn in turns:
            activation = _required_mapping(
                turn, "activation_metadata", trajectory_id
            )
            if activation.get("storage_status") != "materialized":
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "does not have a materialized activation artifact."
                )
            storage_path = _required_string(activation, "storage_path")
            local_path = (root / storage_path).resolve()
            if not local_path.is_relative_to(root):
                raise ValueError(
                    f"Activation path {storage_path!r} escapes artifact_root."
                )
            local_available = local_path.is_file()
            if require_local and not local_available:
                raise FileNotFoundError(
                    f"Activation artifact is not available locally: {local_path}"
                )
            checksum = (
                activation.get("layer_metadata", {}).get("checksum_sha256")
                if isinstance(activation.get("layer_metadata"), dict)
                else None
            )
            if verify_checksums and local_available:
                _verify_checksum(local_path, checksum)

            layers = activation.get("layers_extracted")
            if not isinstance(layers, list) or not layers or not all(
                isinstance(layer, int) for layer in layers
            ):
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "must declare integer layers_extracted."
                )
            checkpoints = activation.get("checkpoint_positions")
            if not isinstance(checkpoints, list) or not checkpoints:
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "must declare checkpoint_positions."
                )
            shapes = activation.get("checkpoint_shapes")
            if not isinstance(shapes, dict):
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "must declare checkpoint_shapes."
                )
            forward_scopes = activation.get("checkpoint_forward_scopes")
            if forward_scopes is None:
                forward_scopes = {}
            if not isinstance(forward_scopes, dict):
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "checkpoint_forward_scopes must be an object when present."
                )
            turn_input = turn.get("input")
            turn_output = turn.get("output")
            if not isinstance(turn_input, dict):
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "must contain input metadata."
                )
            if not isinstance(turn_output, dict):
                raise ValueError(
                    f"Trajectory {trajectory_id!r} step {turn.get('step_index')} "
                    "must contain output metadata."
                )
            rendered_prompt_hash = _required_string(
                turn_input, "rendered_prompt_hash"
            )
            input_token_ids = _required_token_ids(
                turn_input,
                "input_token_ids",
                trajectory_id=trajectory_id,
                step_index=turn.get("step_index"),
            )
            generated_token_ids = _required_token_ids(
                turn_output,
                "generated_token_ids",
                trajectory_id=trajectory_id,
                step_index=turn.get("step_index"),
            )

            for checkpoint_metadata in checkpoints:
                if not isinstance(checkpoint_metadata, dict):
                    raise ValueError("checkpoint_positions entries must be objects.")
                checkpoint = _required_string(checkpoint_metadata, "name")
                if checkpoint not in CHECKPOINT_NAMES:
                    raise ValueError(f"Unsupported activation checkpoint {checkpoint!r}.")
                shape = shapes.get(checkpoint)
                if (
                    not isinstance(shape, list)
                    or len(shape) != 2
                    or shape[0] != len(layers)
                    or not all(isinstance(size, int) and size > 0 for size in shape)
                ):
                    raise ValueError(
                        f"Invalid shape for {trajectory_id!r} step "
                        f"{turn.get('step_index')} checkpoint {checkpoint!r}."
                    )
                key = (trajectory_id, int(turn["step_index"]), checkpoint)
                if key in seen:
                    raise ValueError(f"Duplicate activation-index key {key!r}.")
                seen.add(key)
                rows.append({
                    "schema_version": ACTIVATION_INDEX_SCHEMA,
                    "trajectory_id": trajectory_id,
                    "step_index": int(turn["step_index"]),
                    "agent_id": _required_string(turn, "agent_id"),
                    "agent_role": _required_string(turn, "agent_role"),
                    "hop_index": int(turn["hop_index"]),
                    "model_name": _required_string(model, "model_name"),
                    "model_revision": _required_string(model, "model_revision"),
                    "thinking_mode": _required_string(model, "thinking_mode"),
                    "delegation_depth": _required_string(record, "delegation_depth"),
                    "treatment": _required_string(record, "treatment"),
                    "match_group_id": _required_string(
                        record, "independence_group_id"
                    ),
                    "matched_pair_id": _required_string(record, "matched_pair_id"),
                    "domain_id": _required_string(record, "domain_id"),
                    "task_family_id": _required_string(record, "task_family_id"),
                    "document_set_id": _required_string(record, "document_set_id"),
                    "outcome_class": _required_string(labels, "outcome_class"),
                    "rendered_prompt_hash": rendered_prompt_hash,
                    "input_token_count": len(input_token_ids),
                    "input_token_ids_sha256": _token_ids_sha256(input_token_ids),
                    "generated_token_count": len(generated_token_ids),
                    "generated_token_ids_sha256": _token_ids_sha256(
                        generated_token_ids
                    ),
                    "finish_reason": turn_output.get("finish_reason"),
                    "generation_truncated": turn_output.get("truncated"),
                    "thinking_complete": turn_output.get("thinking_complete"),
                    "labels": {
                        "injection_present": _binary_or_none(
                            labels.get("injection_present"),
                            field="injection_present",
                        ),
                        "unsafe_action_executed": _binary_or_none(
                            action_channel.get("unsafe_action_executed"),
                            field="unsafe_action_executed",
                        ),
                        "output_adoption": _binary_or_none(
                            behavioral_channel.get("output_adoption"),
                            field="output_adoption",
                        ),
                    },
                    "checkpoint": checkpoint,
                    "checkpoint_forward_scope": forward_scopes.get(checkpoint),
                    "checkpoint_position": dict(checkpoint_metadata),
                    "layers": list(layers),
                    "shape": list(shape),
                    "dtype": activation.get("layer_metadata", {}).get("dtype"),
                    "artifact_format_version": activation.get(
                        "artifact_format_version"
                    ),
                    "remote_storage_path": storage_path,
                    "local_path": local_path.as_posix(),
                    "local_available": local_available,
                    "checksum_sha256": checksum,
                })
    return sorted(
        rows,
        key=lambda row: (
            row["thinking_mode"],
            row["match_group_id"],
            row["delegation_depth"],
            row["treatment"],
            row["step_index"],
            row["checkpoint"],
        ),
    )


def write_activation_index(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    """Write index rows as stable JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_activation_index(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate activation-index JSONL rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from error
            if row.get("schema_version") != ACTIVATION_INDEX_SCHEMA:
                raise ValueError(
                    f"Unsupported activation index schema on line {line_number}."
                )
            rows.append(row)
    if not rows:
        raise ValueError("Activation index contains no rows.")
    return rows


def filter_activation_index(
    rows: Iterable[dict[str, Any]],
    *,
    thinking_mode: str | None = None,
    agent_role: str | None = None,
    agent_id: str | None = None,
    checkpoint: str | None = None,
    delegation_depth: str | None = None,
) -> list[dict[str, Any]]:
    """Select a homogeneous probe cohort using explicit metadata fields."""

    expected = {
        "thinking_mode": thinking_mode,
        "agent_role": agent_role,
        "agent_id": agent_id,
        "checkpoint": checkpoint,
        "delegation_depth": delegation_depth,
    }
    return [
        row
        for row in rows
        if all(value is None or row.get(field) == value for field, value in expected.items())
    ]


def load_activation_checkpoint(
    path: str | Path,
    *,
    checkpoint: str = "primary",
    expected_checksum: str | None = None,
    verify_checksum: bool = False,
) -> LoadedActivationCheckpoint:
    """Load one named checkpoint from a current or legacy ``.pt`` artifact.

    Legacy single-position files remain readable through ``checkpoint='primary'``.
    Named checkpoint selection requires the current multi-position format.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Activation artifact does not exist: {source}")
    if verify_checksum:
        _verify_checksum(source, expected_checksum)
    try:
        artifact = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot load activation artifact {source}: {error}") from error
    if not isinstance(artifact, dict):
        raise ValueError(f"Activation artifact {source} must contain a mapping.")

    layers = artifact.get("layers")
    if not isinstance(layers, list) or not layers or not all(
        isinstance(layer, int) for layer in layers
    ):
        raise ValueError(f"Activation artifact {source} has invalid layers metadata.")
    format_version = artifact.get("artifact_format_version")
    if checkpoint == "primary":
        selected_name = str(artifact.get("primary_checkpoint") or "primary")
        tensor = artifact.get("activations")
    else:
        if checkpoint not in CHECKPOINT_NAMES:
            raise ValueError(f"Unsupported activation checkpoint {checkpoint!r}.")
        if format_version != ACTIVATION_ARTIFACT_FORMAT:
            raise ValueError(
                "Named checkpoint selection requires a current multi-position artifact; "
                "use checkpoint='primary' for a legacy single-position artifact."
            )
        positions = artifact.get("position_activations")
        if not isinstance(positions, dict) or checkpoint not in positions:
            raise ValueError(
                f"Activation artifact {source} does not contain checkpoint {checkpoint!r}."
            )
        selected_name = checkpoint
        tensor = positions[checkpoint]

    if not isinstance(tensor, Tensor):
        raise ValueError(f"Activation artifact {source} has no tensor for {checkpoint!r}.")
    tensor = tensor.detach().to(device="cpu")
    if tensor.ndim != 2 or tensor.shape[0] != len(layers):
        raise ValueError(
            f"Activation tensor in {source} must have shape [layers, hidden_size]."
        )
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"Activation tensor in {source} contains NaN or infinity.")
    return LoadedActivationCheckpoint(
        checkpoint=selected_name,
        activations=tensor,
        layers=tuple(layers),
        artifact_format_version=(str(format_version) if format_version else None),
        source_path=source,
    )


def load_probe_activation_batch(
    rows: Sequence[dict[str, Any]],
    *,
    label_target: str,
    layers: Sequence[int] | None = None,
    verify_checksums: bool = False,
) -> ProbeActivationBatch:
    """Load aligned per-layer matrices for one filtered index cohort."""

    if label_target not in LABEL_TARGETS:
        raise ValueError(
            f"label_target must be one of {sorted(LABEL_TARGETS)}, got {label_target!r}."
        )
    if not rows:
        raise ValueError("At least one activation-index row is required.")
    checkpoints = {row.get("checkpoint") for row in rows}
    if len(checkpoints) != 1:
        raise ValueError("Probe cohorts must contain exactly one activation checkpoint.")

    loaded: list[LoadedActivationCheckpoint] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    seen_samples: set[tuple[str, int]] = set()
    for row in rows:
        sample_key = (str(row.get("trajectory_id")), int(row.get("step_index", -1)))
        if sample_key in seen_samples:
            raise ValueError(f"Duplicate probe sample {sample_key!r}.")
        seen_samples.add(sample_key)
        label = row.get("labels", {}).get(label_target)
        if label not in (0, 1):
            raise ValueError(
                f"Probe sample {sample_key!r} has no binary {label_target!r} label."
            )
        loaded.append(load_activation_checkpoint(
            row["local_path"],
            checkpoint=row["checkpoint"],
            expected_checksum=row.get("checksum_sha256"),
            verify_checksum=verify_checksums,
        ))
        labels.append(int(label))
        metadata.append(dict(row))

    common_layers = set(loaded[0].layers)
    for item in loaded[1:]:
        common_layers.intersection_update(item.layers)
    selected_layers = (
        tuple(sorted(common_layers))
        if layers is None
        else tuple(dict.fromkeys(int(layer) for layer in layers))
    )
    missing = sorted(set(selected_layers) - common_layers)
    if missing:
        raise ValueError(f"Requested layers are missing from the cohort: {missing}")
    if not selected_layers:
        raise ValueError("No common activation layers are available for the cohort.")

    matrices: dict[int, np.ndarray] = {}
    for layer in selected_layers:
        vectors = []
        for item in loaded:
            layer_offset = item.layers.index(layer)
            vectors.append(item.activations[layer_offset].float().numpy())
        matrices[layer] = np.stack(vectors).astype(np.float32, copy=False)
    return ProbeActivationBatch(
        activations=matrices,
        labels=np.asarray(labels, dtype=int),
        metadata=metadata,
    )


def summarize_activation_index(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return compact coverage counts for an activation index."""

    def counts(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in rows:
            value = str(row.get(field))
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    unique_turns = {
        (row["trajectory_id"], row["step_index"])
        for row in rows
    }
    return {
        "schema_version": ACTIVATION_INDEX_SCHEMA,
        "index_rows": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "model_turns": len(unique_turns),
        "local_artifacts": len({
            row["local_path"] for row in rows if row.get("local_available")
        }),
        "missing_local_artifacts": len({
            row["local_path"] for row in rows if not row.get("local_available")
        }),
        "thinking_modes": counts("thinking_mode"),
        "checkpoints": counts("checkpoint"),
        "agent_roles": counts("agent_role"),
        "delegation_depths": counts("delegation_depth"),
        "label_positive_counts": {
            target: sum(row["labels"].get(target) == 1 for row in rows)
            for target in sorted(LABEL_TARGETS)
        },
    }


def _required_mapping(
    value: dict[str, Any], field: str, context: str
) -> dict[str, Any]:
    result = value.get(field)
    if not isinstance(result, dict):
        raise ValueError(f"{context!r} must contain an object at {field!r}.")
    return result


def _required_string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"Required string field {field!r} is missing.")
    return result


def _binary_or_none(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if value not in (False, True, 0, 1):
        raise ValueError(f"Label {field!r} must be binary or null.")
    return int(value)


def _required_token_ids(
    value: dict[str, Any],
    field: str,
    *,
    trajectory_id: str,
    step_index: Any,
) -> list[int]:
    token_ids = value.get(field)
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(
            f"Trajectory {trajectory_id!r} step {step_index} must contain "
            f"non-empty {field}."
        )
    if any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(
            f"Trajectory {trajectory_id!r} step {step_index} has invalid {field}."
        )
    return token_ids


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_checksum(path: Path, expected_checksum: str | None) -> None:
    if not isinstance(expected_checksum, str) or not expected_checksum:
        raise ValueError(f"No checksum is declared for {path}.")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected_checksum:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_checksum}, got {observed}."
        )
