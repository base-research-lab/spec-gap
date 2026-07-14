"""Modal runner for Qwen3-32B generation and residual-stream extraction.

The default local action validates a synthetic or real request without calling
Modal compute. A paid H200 run requires the explicit confirmation string
``RUN_H200``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from src.infrastructure.qwen_modal import (
    MODEL_ID,
    MODEL_REVISION,
    activation_artifact_path,
    build_activation_checkpoint_plan,
    build_generation_result,
    build_injection_token_alignment,
    split_thinking_text,
    validate_generation_request,
)
from src.infrastructure.modal_costs import (
    build_gpu_cost_record,
    build_token_usage,
    resolve_modal_input_id,
)


APP_NAME = "spec-gap-qwen3-32b"
MODEL_VOLUME_NAME = "spec-gap-qwen3-32b-model"
ARTIFACT_VOLUME_NAME = "spec-gap-scenario1-artifacts"
MODEL_MOUNT = Path("/models")
ARTIFACT_MOUNT = Path("/artifacts")
MODEL_PATH = MODEL_MOUNT / "Qwen--Qwen3-32B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.8.0",
        "accelerate==1.12.0",
        "huggingface-hub[hf-xet]>=0.34,<2",
        "safetensors>=0.5,<1",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("src")
)

app = modal.App(
    APP_NAME,
    tags={"project": "spec-gap", "component": "qwen3-inference"},
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={MODEL_MOUNT.as_posix(): model_volume},
    cpu=4,
    memory=32768,
    timeout=14400,
)
def download_model(revision: str = MODEL_REVISION) -> dict[str, str]:
    """Download and pin the public model without using a GPU."""

    from huggingface_hub import HfApi, snapshot_download

    resolved_revision = HfApi().model_info(MODEL_ID, revision=revision).sha
    snapshot_download(
        repo_id=MODEL_ID,
        revision=resolved_revision,
        local_dir=MODEL_PATH,
    )
    metadata = {
        "model_id": MODEL_ID,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
    }
    (MODEL_PATH / "spec_gap_model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    model_volume.commit()
    return metadata


@app.cls(
    image=image,
    gpu="H200",
    volumes={
        MODEL_MOUNT.as_posix(): model_volume,
        ARTIFACT_MOUNT.as_posix(): artifact_volume,
    },
    max_containers=1,
    buffer_containers=0,
    scaledown_window=300,
    timeout=3600,
    startup_timeout=1800,
)
class Qwen3Runner:
    """One warm Qwen3-32B instance shared by sequential agent turns."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        metadata_path = MODEL_PATH / "spec_gap_model_metadata.json"
        if not metadata_path.exists():
            raise RuntimeError(
                "Qwen weights are not cached. Run the download action before RUN_H200."
            )
        self.model_metadata = json.loads(metadata_path.read_text())
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        if int(self.model.config.num_hidden_layers) != 64:
            raise RuntimeError(
                f"expected 64 Qwen3-32B layers, got {self.model.config.num_hidden_layers}"
            )

    def _extract_position_activations(
        self,
        full_token_ids,
        layers: list[int],
        checkpoints: list[dict[str, Any]],
    ):
        torch = self.torch
        captured = {}
        handles = []
        positions = [checkpoint["sequence_index"] for checkpoint in checkpoints]

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_index] = hidden[0, positions, :].detach().to(
                    dtype=torch.bfloat16,
                    device="cpu",
                )

            return hook

        for layer in layers:
            handles.append(
                self.model.model.layers[layer].register_forward_hook(make_hook(layer))
            )
        try:
            attention_mask = torch.ones_like(full_token_ids)
            with torch.inference_mode():
                self.model(
                    input_ids=full_token_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(set(layers) - set(captured))
        if missing:
            raise RuntimeError(f"activation hooks did not fire for layers {missing}")
        stacked = torch.stack(
            [captured[layer] for layer in layers], dim=0
        ).transpose(0, 1).contiguous()
        return {
            checkpoint["name"]: stacked[index]
            for index, checkpoint in enumerate(checkpoints)
        }

    @modal.method()
    def generate_agent_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.perf_counter()
        modal_input_id = resolve_modal_input_id(
            modal.current_input_id(),
            modal.current_function_call_id(),
        )
        request = validate_generation_request(payload)
        torch = self.torch
        settings = request["generation_settings"]
        requested_revision = request["model_revision"]
        resolved_revision = self.model_metadata["resolved_revision"]
        if requested_revision not in {"main", resolved_revision}:
            raise ValueError(
                f"request requires model revision {requested_revision!r}, but the "
                f"loaded revision is {resolved_revision!r}"
            )

        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])

        template_kwargs = {
            "conversation": request["messages"],
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": request["enable_thinking"],
        }
        if request["tools"]:
            template_kwargs["tools"] = request["tools"]
        model_inputs = self.tokenizer.apply_chat_template(**template_kwargs)
        input_token_ids = model_inputs["input_ids"][0].tolist()
        rendered_input = self.tokenizer.decode(
            input_token_ids,
            skip_special_tokens=False,
        )
        offset_mapping = None
        if request["injection_text"] is not None:
            if not self.tokenizer.is_fast:
                raise RuntimeError(
                    "injection token alignment requires a fast tokenizer"
                )
            alignment_encoding = self.tokenizer(
                rendered_input,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            if alignment_encoding["input_ids"] != input_token_ids:
                raise RuntimeError(
                    "alignment tokenization does not match the model input IDs"
                )
            offset_mapping = alignment_encoding["offset_mapping"]
        token_alignment = build_injection_token_alignment(
            rendered_input=rendered_input,
            input_token_ids=input_token_ids,
            injection_text=request["injection_text"],
            offset_mapping=offset_mapping,
            tokenizer_name=MODEL_ID,
            tokenizer_revision=resolved_revision,
        )
        model_inputs = {
            key: value.to(self.model.device) for key, value in model_inputs.items()
        }
        input_length = int(model_inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            sequences = self.model.generate(
                **model_inputs,
                do_sample=settings["do_sample"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                min_p=settings["min_p"],
                max_new_tokens=settings["max_new_tokens"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tensor = sequences[0, input_length:]
        generated_ids = generated_tensor.tolist()
        if not generated_ids:
            raise RuntimeError("model returned no generated tokens")

        raw_generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        )
        parsed = split_thinking_text(raw_generated_text)
        close_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        close_index = None
        if request["enable_thinking"] and close_token_id in generated_ids:
            close_index = len(generated_ids) - 1 - generated_ids[::-1].index(
                close_token_id
            )
            thinking_ids = generated_ids[:close_index]
            final_ids = generated_ids[close_index + 1 :]
            thinking_content = self.tokenizer.decode(
                thinking_ids,
                skip_special_tokens=True,
            ).removeprefix("<think>").strip()
            final_content = self.tokenizer.decode(
                final_ids,
                skip_special_tokens=True,
            ).strip()
            thinking_complete = True
            thinking_token_count = len(thinking_ids)
            final_output_token_count = len(final_ids)
        elif request["enable_thinking"]:
            thinking_content = parsed["thinking_content"] or raw_generated_text.strip()
            final_content = ""
            thinking_complete = False
            thinking_token_count = len(generated_ids)
            final_output_token_count = 0
        else:
            thinking_content = None
            final_content = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            thinking_complete = None
            thinking_token_count = 0
            final_output_token_count = len(generated_ids)
        downstream_message = final_content

        eos_id = self.tokenizer.eos_token_id
        stopped = eos_id is not None and eos_id in generated_ids
        finish_reason = "stop" if stopped else "length"
        truncated = finish_reason == "length"

        activation_metadata = None
        if request["extract_activations"]:
            special_token_ids = set(self.tokenizer.all_special_ids)
            if isinstance(close_token_id, int):
                special_token_ids.add(close_token_id)
            checkpoint_plan = build_activation_checkpoint_plan(
                input_token_ids=input_token_ids,
                generated_token_ids=generated_ids,
                enable_thinking=request["enable_thinking"],
                thinking_close_offset=close_index,
                special_token_ids=special_token_ids,
            )
            checkpoints = checkpoint_plan["checkpoints"]
            primary_name = checkpoint_plan["primary_checkpoint"]
            primary_sequence_index = checkpoint_plan["primary_sequence_index"]
            input_checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint["name"] == "last_input_token"
            ]
            generated_checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if checkpoint["name"] != "last_input_token"
            ]
            if len(input_checkpoints) != 1 or not generated_checkpoints:
                raise RuntimeError(
                    "activation plan must contain one input checkpoint and at least "
                    "one generated checkpoint"
                )

            # Isolate the strict input control from the generated continuation.
            # A longer tensor can select a different GPU attention kernel even
            # though later tokens are causally masked from this position.
            position_activations = self._extract_position_activations(
                model_inputs["input_ids"],
                request["activation_layers"],
                input_checkpoints,
            )
            activation_sequence = sequences[:, : primary_sequence_index + 1]
            position_activations.update(self._extract_position_activations(
                activation_sequence,
                request["activation_layers"],
                generated_checkpoints,
            ))
            checkpoint_forward_scopes = {
                "last_input_token": "prompt_only",
                **{
                    checkpoint["name"]: "generated_prefix"
                    for checkpoint in generated_checkpoints
                },
            }
            activation_tensor = position_activations[primary_name]
            by_name = {item["name"]: item for item in checkpoints}
            relative_path = activation_artifact_path(request)
            absolute_path = ARTIFACT_MOUNT / relative_path
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "activations": activation_tensor,
                    "layers": request["activation_layers"],
                    "token_position": request["activation_token_position"],
                    "token_id": by_name[primary_name]["token_id"],
                    "artifact_format_version": checkpoint_plan[
                        "artifact_format_version"
                    ],
                    "primary_checkpoint": primary_name,
                    "position_activations": position_activations,
                    "checkpoint_positions": checkpoints,
                    "checkpoint_forward_scopes": checkpoint_forward_scopes,
                },
                absolute_path,
            )
            checksum = hashlib.sha256(absolute_path.read_bytes()).hexdigest()
            artifact_volume.commit()
            activation_metadata = {
                "storage_status": "materialized",
                "storage_path": relative_path,
                "checksum_sha256": checksum,
                "shape": list(activation_tensor.shape),
                "dtype": str(activation_tensor.dtype),
                "layers": request["activation_layers"],
                "token_position": request["activation_token_position"],
                "token_id": by_name[primary_name]["token_id"],
                "artifact_format_version": checkpoint_plan[
                    "artifact_format_version"
                ],
                "primary_checkpoint": primary_name,
                "checkpoint_positions": checkpoints,
                "checkpoint_forward_scopes": checkpoint_forward_scopes,
                "checkpoint_shapes": {
                    name: list(tensor.shape)
                    for name, tensor in position_activations.items()
                },
                "sequence_length": int(activation_sequence.shape[-1]),
            }

        token_usage = build_token_usage(
            input_token_count=input_length,
            generated_token_count=len(generated_ids),
            thinking_token_count=thinking_token_count,
            final_output_token_count=final_output_token_count,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        elapsed_seconds = time.perf_counter() - started_monotonic
        cost_metadata = build_gpu_cost_record(
            trajectory_id=request["trajectory_id"],
            step_index=request["step_index"],
            agent_id=request["agent_id"],
            thinking_mode=request["thinking_mode"],
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed_seconds,
            modal_input_id=modal_input_id,
            modal_task_id=os.getenv("MODAL_TASK_ID"),
            token_usage=token_usage,
            runtime_metadata={
                "app_name": APP_NAME,
                "model_name": MODEL_ID,
                "model_revision": self.model_metadata["resolved_revision"],
                "modal_environment": os.getenv("MODAL_ENVIRONMENT"),
                "modal_cloud_provider": os.getenv("MODAL_CLOUD_PROVIDER"),
                "modal_region": os.getenv("MODAL_REGION"),
            },
        )
        cost_path = ARTIFACT_MOUNT / cost_metadata["storage_path"]
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(json.dumps(cost_metadata, indent=2) + "\n")
        artifact_volume.commit()
        return build_generation_result(
            request,
            model_revision=self.model_metadata["resolved_revision"],
            tokenizer_name=MODEL_ID,
            tokenizer_revision=self.model_metadata["resolved_revision"],
            rendered_input=rendered_input,
            input_token_ids=input_token_ids,
            generated_token_ids=generated_ids,
            raw_generated_text=raw_generated_text,
            thinking_content=thinking_content,
            thinking_complete=thinking_complete,
            final_content=downstream_message,
            finish_reason=finish_reason,
            truncated=truncated,
            activation_metadata=activation_metadata,
            cost_metadata=cost_metadata,
            token_alignment=token_alignment,
        )


@app.local_entrypoint()
def main(
    request_path: str,
    action: str = "validate",
    output_path: str = "",
    model_revision: str = MODEL_REVISION,
    confirm_paid_run: str = "",
) -> None:
    """Validate by default; remote download and H200 execution are explicit."""

    payload = json.loads(Path(request_path).read_text())
    request = validate_generation_request(payload)

    if action == "validate":
        print(json.dumps(request, indent=2))
        print("Validation passed. No remote function or GPU was started.")
        return
    if action == "download":
        print(json.dumps(download_model.remote(model_revision), indent=2))
        return
    if action != "run":
        raise ValueError("action must be validate, download, or run")
    if confirm_paid_run != "RUN_H200":
        raise ValueError(
            "H200 execution requires --confirm-paid-run RUN_H200"
        )

    result = Qwen3Runner().generate_agent_turn.remote(request)
    rendered = json.dumps(result, indent=2) + "\n"
    if output_path:
        Path(output_path).write_text(rendered)
        print(f"Wrote {output_path}")
    else:
        print(rendered)
