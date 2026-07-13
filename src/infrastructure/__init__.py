"""Infrastructure helpers for remote model execution."""

from .modal_costs import (
    COST_SCHEMA_VERSION,
    GPU_COUNT,
    GPU_TYPE,
    H200_USD_PER_SECOND,
    CostRecordValidationError,
    build_gpu_cost_record,
    build_token_usage,
    cost_record_path,
    validate_gpu_cost_record,
)
from .qwen_modal import (
    CONTROLLED_GENERATION_SETTINGS,
    MODEL_ID,
    MODEL_LAYER_COUNT,
    RequestValidationError,
    build_generation_result,
    generation_result_to_agent_turn_fields,
    parse_tool_call_requests,
    rendered_input_sha256,
    split_thinking_text,
    validate_generation_request,
    validate_generation_result,
)

__all__ = [
    "COST_SCHEMA_VERSION",
    "CONTROLLED_GENERATION_SETTINGS",
    "GPU_COUNT",
    "GPU_TYPE",
    "H200_USD_PER_SECOND",
    "MODEL_ID",
    "MODEL_LAYER_COUNT",
    "CostRecordValidationError",
    "RequestValidationError",
    "build_gpu_cost_record",
    "build_generation_result",
    "build_token_usage",
    "cost_record_path",
    "generation_result_to_agent_turn_fields",
    "parse_tool_call_requests",
    "rendered_input_sha256",
    "split_thinking_text",
    "validate_generation_request",
    "validate_generation_result",
    "validate_gpu_cost_record",
]
