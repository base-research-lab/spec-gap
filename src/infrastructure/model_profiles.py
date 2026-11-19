"""Pinned model profiles for white-box Scenario 1 runs.

A profile fixes everything that is specific to one model: the Hugging Face id
and pinned revision, layer count, how the weights load and where the decoder
blocks live (for the residual-stream hooks), the prespecified probe layers,
and the Modal app, Volumes, and local output folder that keep each model's
runs apart.

Select a profile with ``SPEC_GAP_MODEL_PROFILE`` before running any stage;
the default is ``qwen3-32b``, which reproduces every constant the pipeline
used before profiles existed. The Modal image carries the same variable into
its containers, so local and remote code always resolve the same profile.

    SPEC_GAP_MODEL_PROFILE=qwen3.5-9b python scripts/run_portable_smoke_test.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass


PROFILE_ENV = "SPEC_GAP_MODEL_PROFILE"
DEFAULT_PROFILE_ID = "qwen3-32b"


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    model_id: str
    revision: str
    layer_count: int
    # "causal_lm" -> AutoModelForCausalLM; "image_text_to_text" -> the
    # multimodal class, loaded whole so checkpoint keys map one-to-one.
    loader: str
    # Dotted attribute path from the loaded model to its decoder-block list.
    decoder_layers_path: str
    # Prespecified probe layers at the same relative depths as Qwen3-32B's
    # 40 (62.5%), 32 (50%), and 48 (75%) of 64. Never chosen from AUROC.
    reference_layer: int
    ablation_layers: tuple[int, int]
    # Format the chat template tells the model to use for tool calls.
    tool_call_format: str
    app_name: str
    model_volume_name: str
    artifact_volume_name: str
    model_dir_name: str
    # Subfolder of experiments/scenario1/trajectories; "" is the original root.
    output_subdir: str
    billing_component: str
    gpu: str = "H200"


QWEN3_32B = ModelProfile(
    profile_id="qwen3-32b",
    model_id="Qwen/Qwen3-32B",
    revision="9216db5781bf21249d130ec9da846c4624c16137",
    layer_count=64,
    loader="causal_lm",
    decoder_layers_path="model.layers",
    reference_layer=40,
    ablation_layers=(32, 48),
    tool_call_format="hermes_json",
    app_name="spec-gap-qwen3-32b",
    model_volume_name="spec-gap-qwen3-32b-model",
    artifact_volume_name="spec-gap-scenario1-artifacts",
    model_dir_name="Qwen--Qwen3-32B",
    output_subdir="",
    billing_component="qwen3-inference",
)

# Qwen3.5-9B: 32 layers (24 Gated DeltaNet + 8 gated full attention, one full
# layer every fourth block) behind a vision tower the prompts never use.
# Revision pinned 2026-10-27. The repo ships no generation_config.json and its
# config's eos is <|endoftext|>, so the runner also stops on <|im_end|>.
# Sampling is the shared controlled protocol (0.6 / 0.95 / top-k 20, both
# thinking modes): the model card's recommended thinking-mode "precise coding"
# setting, and the exact Qwen3-32B setting.
# GPU: the thinking-off grid ran on an H200; the 9B model leaves most of an
# H200 idle at batch size 1, so the thinking-on grid (controlled_v2_16384)
# runs on an A100-80GB at 55% of the rate. Each turn records its GPU.
QWEN3_5_9B = ModelProfile(
    profile_id="qwen3.5-9b",
    model_id="Qwen/Qwen3.5-9B",
    revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    layer_count=32,
    loader="image_text_to_text",
    decoder_layers_path="model.language_model.layers",
    reference_layer=20,
    ablation_layers=(16, 24),
    tool_call_format="qwen3_coder_xml",
    app_name="spec-gap-qwen3-5-9b",
    model_volume_name="spec-gap-qwen3-5-9b-model",
    artifact_volume_name="spec-gap-qwen3-5-9b-artifacts",
    model_dir_name="Qwen--Qwen3.5-9B",
    output_subdir="qwen3_5_9b",
    billing_component="qwen3-5-inference",
    gpu="A100-80GB",
)

# Qwen2.5-72B-Instruct: 80 layers, Qwen2ForCausalLM (text-only), hidden 8192.
# Same family as Qwen3-32B at larger scale (Experiment 2). Probe layers scale
# Qwen3-32B's 40/32/48 of 64 to 50/40/60 of 80. bf16 ~145 GB, so device_map
# shards across 2x H200. Revision pinned to "main" for now — pin a commit before
# the official run.
QWEN2_5_72B = ModelProfile(
    profile_id="qwen2.5-72b",
    model_id="Qwen/Qwen2.5-72B-Instruct",
    revision="main",
    layer_count=80,
    loader="causal_lm",
    decoder_layers_path="model.layers",
    reference_layer=50,
    ablation_layers=(40, 60),
    tool_call_format="hermes_json",
    app_name="spec-gap-qwen2-5-72b",
    model_volume_name="spec-gap-qwen2-5-72b-model",
    artifact_volume_name="spec-gap-qwen2-5-72b-artifacts",
    model_dir_name="Qwen--Qwen2.5-72B-Instruct",
    output_subdir="qwen2_5_72b",
    billing_component="qwen2-5-inference",
    gpu="H200:2",
)

# Gemma 4 31B instruction-tuned (google/gemma-4-31B-it, released 2026-04-02):
# a multimodal Gemma4ForConditionalGeneration (text + vision). The prompts are
# text-only, so it loads whole via AutoModelForImageTextToText with the vision
# tower idle, exactly like Qwen3.5-9B. The TEXT decoder has 60 layers
# (text_config.num_hidden_layers), hidden 5376; probe layers scale Qwen3-32B's
# 40/32/48 of 64 to 37/30/45 of 60. bf16 ~62 GB fits one H200. Different family
# from Qwen (Experiment 2's cross-family test). VALIDATE on first load:
#   1. decoder_layers_path resolves to the 60-block text decoder (the runner
#      asserts len == layer_count);
#   2. tool-call parsing handles Gemma's format — the scorer's
#      parse_tool_call_requests may need a Gemma branch, or executions undercount;
#   3. pin a revision before the official run.
GEMMA4_31B = ModelProfile(
    profile_id="gemma4-31b",
    model_id="google/gemma-4-31B-it",
    revision="main",
    layer_count=60,
    loader="image_text_to_text",
    decoder_layers_path="model.language_model.layers",
    reference_layer=37,
    ablation_layers=(30, 45),
    tool_call_format="gemma_tool_call",
    app_name="spec-gap-gemma4-31b",
    model_volume_name="spec-gap-gemma4-31b-model",
    artifact_volume_name="spec-gap-gemma4-31b-artifacts",
    model_dir_name="google--gemma-4-31B-it",
    output_subdir="gemma4_31b",
    billing_component="gemma4-inference",
    gpu="H200",
)

PROFILES = {
    profile.profile_id: profile
    for profile in (QWEN3_32B, QWEN3_5_9B, QWEN2_5_72B, GEMMA4_31B)
}


def get_profile(profile_id: str | None = None) -> ModelProfile:
    selected = (profile_id or DEFAULT_PROFILE_ID).strip()
    try:
        return PROFILES[selected]
    except KeyError:
        raise ValueError(
            f"unknown model profile {selected!r}; set {PROFILE_ENV} to one of "
            f"{sorted(PROFILES)}"
        ) from None


ACTIVE_PROFILE = get_profile(os.environ.get(PROFILE_ENV))


def scale_qwen3_layers(
    layers: tuple[int, ...] | list[int],
    profile: ModelProfile = ACTIVE_PROFILE,
) -> tuple[int, ...]:
    """Map Qwen3-32B layer indices (0-63) to the same relative depth.

    Identity for Qwen3-32B; for Qwen3.5-9B it maps 32/40/48 to 16/20/24, the
    profile's prespecified layers, and 63 to the last layer.
    """

    span = QWEN3_32B.layer_count - 1
    return tuple(
        round(layer * (profile.layer_count - 1) / span) for layer in layers
    )
