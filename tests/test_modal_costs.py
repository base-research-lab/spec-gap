from copy import deepcopy

import pytest

from src.infrastructure.modal_costs import (
    H200_USD_PER_SECOND,
    CostRecordValidationError,
    build_gpu_cost_record,
    build_token_usage,
    cost_record_path,
    validate_gpu_cost_record,
)


def token_usage():
    return build_token_usage(
        input_token_count=100,
        generated_token_count=120,
        thinking_token_count=80,
        final_output_token_count=39,
    )


def cost_record():
    return build_gpu_cost_record(
        trajectory_id="group01_injected_3hop_thinking_on",
        step_index=2,
        agent_id="worker_2",
        thinking_mode="on",
        started_at="2026-07-13T12:00:00+00:00",
        finished_at="2026-07-13T12:01:00+00:00",
        elapsed_seconds=60.0,
        modal_input_id="in-123",
        modal_task_id="ta-456",
        token_usage=token_usage(),
        runtime_metadata={"modal_region": "us-east-1"},
    )


def test_token_usage_partitions_thinking_final_and_separator_tokens():
    usage = token_usage()

    assert usage == {
        "input_tokens": 100,
        "generated_tokens": 120,
        "thinking_tokens": 80,
        "final_output_tokens": 39,
        "special_or_separator_tokens": 1,
        "total_tokens_processed": 220,
    }


def test_token_usage_rejects_an_impossible_partition():
    with pytest.raises(CostRecordValidationError, match="cannot exceed"):
        build_token_usage(
            input_token_count=10,
            generated_token_count=5,
            thinking_token_count=4,
            final_output_token_count=2,
        )


def test_one_h200_minute_has_a_pinned_reproducible_estimate():
    record = cost_record()

    assert H200_USD_PER_SECOND == 0.001261
    assert record["gpu_type"] == "H200"
    assert record["gpu_count"] == 1
    assert record["estimated_h200_cost_usd"] == 0.07566
    assert record["generated_tokens_per_second"] == 2.0
    assert record["estimated_h200_cost_per_1k_generated_tokens_usd"] == 0.6305
    assert record["billing_status"] == "estimate_pending_workspace_reconciliation"


def test_cost_path_is_scoped_by_trajectory_mode_step_and_input():
    assert cost_record_path(
        trajectory_id="trajectory-1",
        thinking_mode="off",
        step_index=3,
        modal_input_id="in-abc",
    ) == "costs/trajectory-1/off/step_003/in-abc.json"


def test_cost_record_lists_what_the_estimate_excludes():
    record = cost_record()

    assert any("startup" in item for item in record["excludes"])
    assert any("CPU" in item for item in record["excludes"])
    assert any("credits" in item for item in record["excludes"])


def test_cost_validation_rejects_tampered_cost_or_path():
    bad_cost = deepcopy(cost_record())
    bad_cost["estimated_h200_cost_usd"] = 99.0
    with pytest.raises(CostRecordValidationError, match="inconsistent"):
        validate_gpu_cost_record(bad_cost)

    bad_path = deepcopy(cost_record())
    bad_path["storage_path"] = "costs/wrong.json"
    with pytest.raises(CostRecordValidationError, match="storage_path"):
        validate_gpu_cost_record(bad_path)
