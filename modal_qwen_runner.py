"""Modal runner for Qwen3-32B generation and residual-stream extraction.

The default local action validates a synthetic or real request without calling
Modal compute. A paid H200 run requires the explicit confirmation string
``RUN_H200``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal

from src.infrastructure.qwen_modal import (
    MODEL_ID,
    activation_artifact_path,
    split_thinking_text,
    validate_generation_request,
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

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    volumes={MODEL_MOUNT.as_posix(): model_volume},
    cpu=4,
    memory=32768,
    timeout=14400,
)
def download_model(revision: str = "main") -> dict[str, str]:
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

    def _last_non_special_offset(self, generated_ids: list[int]) -> int:
        special = set(self.tokenizer.all_special_ids)
        for offset in range(len(generated_ids) - 1, -1, -1):
            if generated_ids[offset] not in special:
                return offset
        return max(len(generated_ids) - 1, 0)

    def _extract_last_token_activations(
        self,
        full_token_ids,
        layers: list[int],
    ):
        torch = self.torch
        captured = {}
        handles = []

        def make_hook(layer_index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_index] = hidden[0, -1, :].detach().to(
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
        return torch.stack([captured[layer] for layer in layers], dim=0)

    @modal.method()
    def generate_agent_turn(self, payload: dict[str, Any]) -> dict[str, Any]:
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
        elif request["enable_thinking"]:
            thinking_content = parsed["thinking_content"] or raw_generated_text.strip()
            final_content = ""
            thinking_complete = False
        else:
            thinking_content = None
            final_content = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
            thinking_complete = None
        downstream_message = final_content

        eos_id = self.tokenizer.eos_token_id
        stopped = eos_id is not None and eos_id in generated_ids
        finish_reason = "stop" if stopped else "length"
        truncated = finish_reason == "length"

        activation_metadata = None
        if request["extract_activations"]:
            content_offset = self._last_non_special_offset(generated_ids)
            activation_sequence = sequences[:, : input_length + content_offset + 1]
            activation_tensor = self._extract_last_token_activations(
                activation_sequence,
                request["activation_layers"],
            )
            relative_path = activation_artifact_path(request)
            absolute_path = ARTIFACT_MOUNT / relative_path
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "activations": activation_tensor,
                    "layers": request["activation_layers"],
                    "token_position": request["activation_token_position"],
                    "token_id": generated_ids[content_offset],
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
                "token_id": generated_ids[content_offset],
            }

        return {
            "trajectory_id": request["trajectory_id"],
            "step_index": request["step_index"],
            "agent_id": request["agent_id"],
            "agent_role": request["agent_role"],
            "hop_index": request["hop_index"],
            "model_name": MODEL_ID,
            "model_revision": self.model_metadata["resolved_revision"],
            "thinking_mode": request["thinking_mode"],
            "generation_settings": settings,
            "rendered_input": self.tokenizer.decode(
                model_inputs["input_ids"][0], skip_special_tokens=False
            ),
            "input_token_ids": model_inputs["input_ids"][0].tolist(),
            "generated_token_ids": generated_ids,
            "raw_generated_text": raw_generated_text,
            "thinking_content": thinking_content,
            "thinking_complete": thinking_complete,
            "final_content": final_content,
            "downstream_message": downstream_message,
            "finish_reason": finish_reason,
            "truncated": truncated,
            "activation_metadata": activation_metadata,
        }


@app.local_entrypoint()
def main(
    request_path: str,
    action: str = "validate",
    output_path: str = "",
    model_revision: str = "main",
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
