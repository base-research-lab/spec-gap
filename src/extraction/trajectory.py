"""Convert trajectory JSONL records into activation-extraction examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from torch import Tensor


DEFAULT_LAYERS = (20, 16, 24)
NEGATIVE_STEP_LABELS = {"task_preserved", "resisted_injection"}
POSITIVE_STEP_LABELS = {
    "compromised_context",
    "unsafe_action_attempted",
    "unsafe_action_executed",
}


@dataclass(frozen=True)
class TrajectoryActivationExample:
    trajectory_id: str
    step_index: int
    node_id: str
    role: str
    model: str
    messages: list[dict]
    step_label: str | None
    binary_label: int | None
    scenario_id: str | None
    condition: str | None
    hop_mode: str
    token_position: str
    trajectory_label: str | None
    failure_mode: str | None
    injection_wording_id: str | None
    contrast_pair_id: str | None
    source_metadata: dict


@dataclass(frozen=True)
class TrajectoryActivationRequest:
    """Model-free extraction request produced from one trajectory step."""

    trajectory_id: str
    step_index: int
    node_id: str
    role: str
    model: str
    rendered_prompt: str
    token_position: str
    step_label: str | None
    binary_label: int | None
    metadata: dict


@dataclass(frozen=True)
class TrajectoryActivationBatch:
    activations: dict[int, Tensor]
    labels: np.ndarray
    metadata: list[dict]
    rendered_prompts: list[str]


def step_label_to_binary(step_label: str | None) -> int | None:
    if step_label in NEGATIVE_STEP_LABELS:
        return 0
    if step_label in POSITIVE_STEP_LABELS:
        return 1
    return None


def records_to_activation_examples(
    records: list[dict],
    *,
    include_unlabeled: bool = False,
) -> list[TrajectoryActivationExample]:
    """Create one extraction example per logged model step."""

    examples = []
    for record in records:
        if record.get("role") == "guard" or str(record.get("node_id", "")).startswith("__"):
            continue
        output = str(record.get("output_message") or "").strip()
        if not output:
            raise ValueError(
                f"Trajectory {record.get('trajectory_id')} step "
                f"{record.get('step_index')} has no output_message."
            )
        input_context = record.get("input_context")
        if not isinstance(input_context, list) or not input_context:
            raise ValueError(
                f"Trajectory {record.get('trajectory_id')} step "
                f"{record.get('step_index')} has no input_context."
            )
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in input_context
        ]
        messages.append({"role": "assistant", "content": output})
        step_label = record.get("step_label")
        binary_label = step_label_to_binary(step_label)
        if binary_label is None and not include_unlabeled:
            continue
        examples.append(TrajectoryActivationExample(
            trajectory_id=str(record["trajectory_id"]),
            step_index=int(record["step_index"]),
            node_id=str(record["node_id"]),
            role=str(record["role"]),
            model=str(record["model"]),
            messages=messages,
            step_label=step_label,
            binary_label=binary_label,
            scenario_id=record.get("scenario_id"),
            condition=record.get("condition"),
            hop_mode=str(record.get("hop_mode") or ""),
            token_position=str(record.get("token_position") or "last"),
            trajectory_label=record.get("trajectory_label"),
            failure_mode=record.get("failure_mode"),
            injection_wording_id=record.get("injection_wording_id"),
            contrast_pair_id=record.get("contrast_pair_id"),
            source_metadata={
                key: record[key]
                for key in (
                    "source_schema_version",
                    "agent_id",
                    "hop_index",
                    "raw_poison_exposed_agents",
                    "activation_metadata",
                    "attention_metadata",
                    "token_alignment",
                    "behavioral_compromise_label",
                    "reasoning_compromise_label",
                )
                if key in record
            },
        ))
    return examples


def render_messages(messages: list[dict], tokenizer) -> str:
    """Render messages with the model template, with a test-model fallback."""

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )


class _PlainTextTokenizer:
    chat_template = None


def records_to_activation_requests(
    records: list[dict],
    *,
    include_unlabeled: bool = False,
    tokenizer=None,
    renderer: Callable[[list[dict], object], str] = render_messages,
) -> list[TrajectoryActivationRequest]:
    """Build prompt strings and metadata without loading a model.

    This is the handoff adapter for new trajectory files: JSONL records can be
    validated and rendered into extraction-ready requests before Llama weights
    or TransformerLens are loaded.
    """

    tokenizer = tokenizer or _PlainTextTokenizer()
    requests = []
    for example in records_to_activation_examples(
        records,
        include_unlabeled=include_unlabeled,
    ):
        metadata = {
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "node_id": example.node_id,
            "role": example.role,
            "model": example.model,
            "step_label": example.step_label,
            "binary_label": example.binary_label,
            "scenario_id": example.scenario_id,
            "condition": example.condition,
            "hop_mode": example.hop_mode,
            "token_position": example.token_position,
            "trajectory_label": example.trajectory_label,
            "failure_mode": example.failure_mode,
            "injection_wording_id": example.injection_wording_id,
            "contrast_pair_id": example.contrast_pair_id,
        }
        metadata.update(example.source_metadata)
        requests.append(TrajectoryActivationRequest(
            trajectory_id=example.trajectory_id,
            step_index=example.step_index,
            node_id=example.node_id,
            role=example.role,
            model=example.model,
            rendered_prompt=renderer(example.messages, tokenizer),
            token_position=example.token_position,
            step_label=example.step_label,
            binary_label=example.binary_label,
            metadata=metadata,
        ))
    return requests


def extract_trajectory_activations(
    model,
    records: list[dict],
    *,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    batch_size: int = 1,
    include_unlabeled: bool = False,
    renderer: Callable[[list[dict], object], str] = render_messages,
) -> TrajectoryActivationBatch:
    """Extract last-output-token activations aligned to trajectory records."""

    examples = records_to_activation_examples(
        records,
        include_unlabeled=include_unlabeled,
    )
    if not examples:
        raise ValueError("No eligible trajectory steps were available for extraction.")

    prompts = [renderer(example.messages, model.tokenizer) for example in examples]
    from src.extraction.residual_stream import extract_residual_stream

    activations = extract_residual_stream(
        model,
        prompts,
        layers=layers,
        token_position="last",
        batch_size=batch_size,
    )
    labels = np.asarray([
        -1 if example.binary_label is None else example.binary_label
        for example in examples
    ], dtype=int)
    metadata = []
    for example in examples:
        row = {
            "trajectory_id": example.trajectory_id,
            "step_index": example.step_index,
            "node_id": example.node_id,
            "role": example.role,
            "model": example.model,
            "step_label": example.step_label,
            "scenario_id": example.scenario_id,
            "condition": example.condition,
            "hop_mode": example.hop_mode,
            "token_position": example.token_position,
            "trajectory_label": example.trajectory_label,
            "failure_mode": example.failure_mode,
            "injection_wording_id": example.injection_wording_id,
            "contrast_pair_id": example.contrast_pair_id,
        }
        row.update(example.source_metadata)
        metadata.append(row)
    return TrajectoryActivationBatch(
        activations=activations,
        labels=labels,
        metadata=metadata,
        rendered_prompts=prompts,
    )
