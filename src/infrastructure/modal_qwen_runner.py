"""Modal runner for Qwen3-32B generation and residual-stream extraction.

The default local action validates a synthetic or real request without calling
Modal compute. A paid H200 run requires the explicit confirmation string
``RUN_H200``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
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
    initialize_torch_sampling_rng,
    split_thinking_text,
    validate_context_window,
    validate_generation_request,
)
from src.infrastructure.modal_billing import (
    BASE_BILLING_TAGS,
    validate_analysis_tier,
)
from src.infrastructure.modal_costs import (
    build_gpu_cost_record,
    build_token_usage,
    resolve_modal_input_id,
)
from src.scenario1.activation_repair import (
    ACTIVATION_REPAIR_METHOD,
    ACTIVATION_REPAIR_RESULT_SCHEMA,
    activation_repair_backup_path,
    validate_activation_repair_request,
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
    tags=BASE_BILLING_TAGS,
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
    def repair_prompt_activation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Replace only the saved last-input state with a prompt-only forward."""

        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.perf_counter()
        modal_input_id = resolve_modal_input_id(
            modal.current_input_id(),
            modal.current_function_call_id(),
        )
        request = validate_activation_repair_request(payload)
        if self.model_metadata["resolved_revision"] != request["model_revision"]:
            raise ValueError("repair request does not match the loaded model revision")

        torch = self.torch
        artifact_path = ARTIFACT_MOUNT / request["storage_path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"activation artifact is missing from the Volume: {artifact_path}"
            )
        existing_bytes = artifact_path.read_bytes()
        existing_checksum = hashlib.sha256(existing_bytes).hexdigest()
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(artifact, dict):
            raise ValueError("activation artifact must contain an object")
        expected_scopes = request["expected_checkpoint_forward_scopes"]
        existing_scopes = artifact.get("checkpoint_forward_scopes")
        if existing_scopes not in (None, {}, expected_scopes):
            raise ValueError("artifact checkpoint scopes are partial or inconsistent")
        status = "already_repaired" if existing_scopes == expected_scopes else "repaired"

        if artifact.get("artifact_format_version") != request[
            "artifact_format_version"
        ]:
            raise ValueError("artifact format does not match the repair request")
        if artifact.get("layers") != request["layers"]:
            raise ValueError("artifact layers do not match the repair request")
        if artifact.get("token_position") != request["token_position"]:
            raise ValueError("artifact token position does not match")
        if artifact.get("token_id") != request["token_id"]:
            raise ValueError("artifact primary token ID does not match")
        if artifact.get("primary_checkpoint") != request["primary_checkpoint"]:
            raise ValueError("artifact primary checkpoint does not match")
        if artifact.get("checkpoint_positions") != request[
            "checkpoint_positions"
        ]:
            raise ValueError("artifact checkpoint positions do not match")
        positions = artifact.get("position_activations")
        if not isinstance(positions, dict) or set(positions) != set(
            expected_scopes
        ):
            raise ValueError(
                "artifact position tensors do not cover every checkpoint"
            )
        for name, tensor in positions.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or list(tensor.shape) != request["checkpoint_shapes"][name]
                or tensor.dtype != torch.bfloat16
                or not bool(torch.isfinite(tensor.float()).all())
            ):
                raise ValueError(f"artifact checkpoint {name!r} is inconsistent")
        primary_tensor = positions[request["primary_checkpoint"]]
        if not isinstance(artifact.get("activations"), torch.Tensor) or not torch.equal(
            artifact["activations"], primary_tensor
        ):
            raise ValueError("artifact primary activation tensor is inconsistent")

        if status == "repaired":
            if existing_checksum != request["expected_checksum_sha256"]:
                raise ValueError(
                    "Volume artifact checksum does not match the saved model turn"
                )
            backup_storage_path = activation_repair_backup_path(
                request["storage_path"]
            )
            backup_path = ARTIFACT_MOUNT / backup_storage_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if backup_path.exists():
                backup_checksum = hashlib.sha256(backup_path.read_bytes()).hexdigest()
                if backup_checksum != request["expected_checksum_sha256"]:
                    raise ValueError("existing Volume backup has the wrong checksum")
            else:
                backup_path.write_bytes(existing_bytes)

            input_checkpoint = next(
                checkpoint
                for checkpoint in request["checkpoint_positions"]
                if checkpoint["name"] == "last_input_token"
            )
            input_ids = torch.tensor(
                [request["input_token_ids"]],
                dtype=torch.long,
                device=self.model.device,
            )
            prompt_activations = self._extract_position_activations(
                input_ids,
                request["layers"],
                [input_checkpoint],
            )["last_input_token"]
            if list(prompt_activations.shape) != request["checkpoint_shapes"][
                "last_input_token"
            ]:
                raise RuntimeError("prompt-only activation shape is inconsistent")
            generated_tensors = {
                name: tensor.clone()
                for name, tensor in positions.items()
                if name != "last_input_token"
            }

            repaired_at = datetime.now(timezone.utc).isoformat()
            repair_metadata = {
                "schema_version": ACTIVATION_REPAIR_RESULT_SCHEMA,
                "repair_method": ACTIVATION_REPAIR_METHOD,
                "repaired_at": repaired_at,
                "model_revision": request["model_revision"],
                "input_token_ids_sha256": request["input_token_ids_sha256"],
                "rendered_input_sha256": request["rendered_input_sha256"],
                "previous_checksum_sha256": request[
                    "expected_checksum_sha256"
                ],
                "replaced_checkpoint": "last_input_token",
                "generated_outputs_preserved": True,
                "generated_checkpoint_tensors_preserved": True,
            }
            artifact["position_activations"][
                "last_input_token"
            ] = prompt_activations
            artifact["checkpoint_forward_scopes"] = expected_scopes
            artifact["activation_repair_metadata"] = repair_metadata
            temporary_path = artifact_path.with_name(
                f".{artifact_path.name}.{modal_input_id}.tmp"
            )
            torch.save(artifact, temporary_path)
            verified = torch.load(
                temporary_path,
                map_location="cpu",
                weights_only=True,
            )
            if (
                verified.get("checkpoint_forward_scopes") != expected_scopes
                or not torch.equal(
                    verified["position_activations"]["last_input_token"],
                    prompt_activations,
                )
            ):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError("saved repair artifact failed verification")
            for name, tensor in generated_tensors.items():
                if not torch.equal(
                    verified["position_activations"][name],
                    tensor,
                ):
                    temporary_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"repair changed generated checkpoint {name!r}"
                    )
            if not torch.equal(verified["activations"], primary_tensor):
                temporary_path.unlink(missing_ok=True)
                raise RuntimeError("repair changed the primary activation tensor")
            os.replace(temporary_path, artifact_path)
            artifact_volume.commit()
        else:
            backup_storage_path = activation_repair_backup_path(
                request["storage_path"]
            )
            repair_metadata = artifact.get("activation_repair_metadata")
            if not isinstance(repair_metadata, dict):
                raise ValueError(
                    "already-repaired artifact is missing repair provenance"
                )
            if repair_metadata.get("previous_checksum_sha256") != request[
                "expected_checksum_sha256"
            ]:
                raise ValueError(
                    "already-repaired artifact does not match the requested origin"
                )
            backup_path = ARTIFACT_MOUNT / backup_storage_path
            if (
                not backup_path.is_file()
                or hashlib.sha256(backup_path.read_bytes()).hexdigest()
                != request["expected_checksum_sha256"]
            ):
                raise ValueError("already-repaired artifact has no valid backup")

        repaired_bytes = artifact_path.read_bytes()
        new_checksum = hashlib.sha256(repaired_bytes).hexdigest()
        primary_tensor = artifact["position_activations"][
            request["primary_checkpoint"]
        ]
        activation_metadata = {
            "storage_status": "materialized",
            "storage_path": request["storage_path"],
            "checksum_sha256": new_checksum,
            "shape": list(primary_tensor.shape),
            "dtype": str(primary_tensor.dtype),
            "layers": request["layers"],
            "token_position": request["token_position"],
            "token_id": request["token_id"],
            "artifact_format_version": request["artifact_format_version"],
            "primary_checkpoint": request["primary_checkpoint"],
            "checkpoint_positions": request["checkpoint_positions"],
            "checkpoint_forward_scopes": expected_scopes,
            "checkpoint_shapes": request["checkpoint_shapes"],
            "sequence_length": request["sequence_length"],
        }
        token_usage = build_token_usage(
            input_token_count=len(request["input_token_ids"]),
            generated_token_count=0,
            thinking_token_count=0,
            final_output_token_count=0,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        cost_metadata = build_gpu_cost_record(
            trajectory_id=request["trajectory_id"],
            step_index=request["step_index"],
            agent_id=request["agent_id"],
            thinking_mode=request["thinking_mode"],
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=time.perf_counter() - started_monotonic,
            modal_input_id=modal_input_id,
            modal_task_id=os.getenv("MODAL_TASK_ID"),
            token_usage=token_usage,
            runtime_metadata={
                "operation": ACTIVATION_REPAIR_METHOD,
                "app_name": APP_NAME,
                "model_name": MODEL_ID,
                "model_revision": self.model_metadata["resolved_revision"],
                "analysis_tier": request.get("analysis_tier"),
                "modal_environment": os.getenv("MODAL_ENVIRONMENT"),
                "modal_cloud_provider": os.getenv("MODAL_CLOUD_PROVIDER"),
                "modal_region": os.getenv("MODAL_REGION"),
            },
        )
        cost_path = ARTIFACT_MOUNT / cost_metadata["storage_path"]
        cost_path.parent.mkdir(parents=True, exist_ok=True)
        cost_path.write_text(json.dumps(cost_metadata, indent=2) + "\n")
        artifact_volume.commit()
        return {
            "schema_version": ACTIVATION_REPAIR_RESULT_SCHEMA,
            "status": status,
            "trajectory_id": request["trajectory_id"],
            "step_index": request["step_index"],
            "agent_id": request["agent_id"],
            "thinking_mode": request["thinking_mode"],
            "new_checksum_sha256": new_checksum,
            "backup_storage_path": backup_storage_path,
            "activation_metadata": activation_metadata,
            "repair_metadata": repair_metadata,
            "cost_metadata": cost_metadata,
            "artifact_bytes": repaired_bytes,
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

        initialize_torch_sampling_rng(torch, settings["seed"])

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
        context_window = getattr(
            self.model.config,
            "max_position_embeddings",
            None,
        )
        if (
            not isinstance(context_window, int)
            or isinstance(context_window, bool)
            or context_window < 1
        ):
            raise RuntimeError(
                "loaded model does not expose a valid max_position_embeddings"
            )
        validate_context_window(
            input_tokens=input_length,
            max_new_tokens=settings["max_new_tokens"],
            context_window_tokens=context_window,
        )

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
                "analysis_tier": request["analysis_tier"],
                "torch_version": str(torch.__version__),
                "transformers_version": importlib.metadata.version(
                    "transformers"
                ),
                "cuda_version": str(torch.version.cuda),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "deterministic_algorithms_enabled": (
                    torch.are_deterministic_algorithms_enabled()
                ),
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
    analysis_tier: str = "exploratory",
) -> None:
    """Validate by default; remote download and H200 execution are explicit."""

    analysis_tier = validate_analysis_tier(analysis_tier)
    payload = json.loads(Path(request_path).read_text())
    payload["analysis_tier"] = analysis_tier
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

    app.set_tags({
        **BASE_BILLING_TAGS,
        "run_kind": "single_model_turn",
        "analysis_tier": analysis_tier,
    })
    result = Qwen3Runner().generate_agent_turn.remote(request)
    rendered = json.dumps(result, indent=2) + "\n"
    if output_path:
        Path(output_path).write_text(rendered)
        print(f"Wrote {output_path}")
    else:
        print(rendered)
