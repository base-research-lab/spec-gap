"""
Tests for residual stream extraction.

These tests use a small model (gpt2) to validate shapes and behavior
without requiring Llama 3 8B weights. The real model is tested via
the sanity check experiment.
"""

import pytest
import torch

from src.extraction.residual_stream import (
    HIDDEN_DIM,
    _get_cache_filter,
    extract_residual_stream,
    load_model,
)
from src.extraction.trajectory import (
    extract_trajectory_activations,
    records_to_activation_examples,
    step_label_to_binary,
)

GPT2_LAYERS = (2, 6, 10)
GPT2_HIDDEN = 768


@pytest.fixture(scope="module")
def gpt2_model():
    return load_model(model_name="gpt2", device="cpu", dtype=torch.float32)


class TestCacheFilter:
    def test_accepts_target_layers(self):
        f = _get_cache_filter((16, 20, 24))
        assert f("blocks.20.hook_resid_post") is True
        assert f("blocks.16.hook_resid_post") is True

    def test_rejects_other_layers(self):
        f = _get_cache_filter((16, 20, 24))
        assert f("blocks.0.hook_resid_post") is False
        assert f("blocks.10.hook_resid_post") is False

    def test_rejects_non_resid_hooks(self):
        f = _get_cache_filter((20,))
        assert f("blocks.20.attn.hook_result") is False
        assert f("blocks.20.hook_resid_pre") is False


class TestExtraction:
    def test_output_shape_last_token(self, gpt2_model):
        prompts = ["Hello world", "This is a test"]
        result = extract_residual_stream(
            gpt2_model, prompts, layers=GPT2_LAYERS, token_position="last",
        )
        for layer in GPT2_LAYERS:
            assert layer in result
            assert result[layer].shape == (2, GPT2_HIDDEN)

    def test_output_shape_all_tokens(self, gpt2_model):
        prompts = ["Hello"]
        result = extract_residual_stream(
            gpt2_model, prompts, layers=(6,), token_position="all",
        )
        assert result[6].ndim == 3
        assert result[6].shape[0] == 1
        assert result[6].shape[2] == GPT2_HIDDEN

    def test_output_shape_int_position(self, gpt2_model):
        prompts = ["Hello world"]
        result = extract_residual_stream(
            gpt2_model, prompts, layers=(6,), token_position=0,
        )
        assert result[6].shape == (1, GPT2_HIDDEN)

    def test_no_nans(self, gpt2_model):
        prompts = ["Test prompt for NaN check"]
        result = extract_residual_stream(
            gpt2_model, prompts, layers=GPT2_LAYERS,
        )
        for layer in GPT2_LAYERS:
            assert not torch.isnan(result[layer]).any()
            assert not torch.isinf(result[layer]).any()

    def test_deterministic(self, gpt2_model):
        prompts = ["Determinism check"]
        r1 = extract_residual_stream(gpt2_model, prompts, layers=(6,))
        r2 = extract_residual_stream(gpt2_model, prompts, layers=(6,))
        assert torch.allclose(r1[6], r2[6])

    def test_different_prompts_give_different_activations(self, gpt2_model):
        prompts = ["The sky is blue", "Cats can fly to Mars"]
        result = extract_residual_stream(
            gpt2_model, prompts, layers=(6,),
        )
        assert not torch.allclose(result[6][0], result[6][1])

    def test_invalid_token_position_raises(self, gpt2_model):
        with pytest.raises(ValueError):
            extract_residual_stream(
                gpt2_model, ["test"], layers=(6,), token_position="middle",
            )

    def test_batching_matches_single(self, gpt2_model):
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        single = extract_residual_stream(
            gpt2_model, prompts, layers=(6,), batch_size=1,
        )
        batched = extract_residual_stream(
            gpt2_model, prompts, layers=(6,), batch_size=3,
        )
        assert torch.allclose(single[6], batched[6], atol=1e-3)


class TestTrajectoryExtraction:
    @staticmethod
    def _records():
        base = {
            "trajectory_id": "synthetic-trajectory",
            "model": "gpt2",
            "scenario_id": "scenario1",
            "condition": "synthetic-2hop",
            "hop_mode": "2-hop",
        }
        return [
            {
                **base,
                "step_index": 0,
                "node_id": "planner",
                "role": "planner",
                "input_context": [{"role": "user", "content": "Summarize the paper."}],
                "output_message": "Ask the worker to retrieve and summarize it.",
                "step_label": "task_preserved",
                "binary_label": 0,
                "label_target": "trajectory_action_executed",
                "contrast_pair_id": "task01-planner-2hop",
            },
            {
                **base,
                "step_index": 1,
                "node_id": "worker",
                "role": "worker",
                "input_context": [
                    {"role": "user", "content": "Summarize the paper."},
                    {"role": "user", "content": "Retrieved document contains an unrelated instruction."},
                ],
                "output_message": "I will follow the unrelated embedded instruction.",
                "step_label": "compromised_context",
                "binary_label": 1,
                "label_target": "trajectory_action_executed",
                "contrast_pair_id": "task01-worker-2hop",
            },
        ]

    def test_step_label_mapping_is_fail_closed(self):
        assert step_label_to_binary("task_preserved") == 0
        assert step_label_to_binary("compromised_context") is None
        assert step_label_to_binary("propagated_but_not_executed") == 0
        assert step_label_to_binary("suspicious_instruction_propagation") is None
        assert step_label_to_binary(None) is None

    def test_records_align_messages_and_labels(self):
        examples = records_to_activation_examples(self._records())
        assert [example.binary_label for example in examples] == [0, 1]
        assert examples[1].messages[-1]["role"] == "assistant"
        assert examples[1].messages[-1]["content"].startswith("I will follow")

    def test_synthetic_trajectory_to_activation(self, gpt2_model):
        batch = extract_trajectory_activations(
            gpt2_model,
            self._records(),
            layers=(6,),
            batch_size=2,
        )
        assert batch.activations[6].shape == (2, GPT2_HIDDEN)
        assert batch.labels.tolist() == [0, 1]
        assert [row["step_index"] for row in batch.metadata] == [0, 1]
        assert all(batch.rendered_prompts)
