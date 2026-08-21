#!/usr/bin/env python3
"""Check retrieved Scenario 1 prompts with Qwen's exact chat template.

This local check uses no model weights and starts no Modal or GPU job. The
remote runner repeats the authoritative check after rendering every real turn.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.qwen_modal import MODEL_ID, MODEL_REVISION  # noqa: E402
from src.scenario1 import generator  # noqa: E402
from src.scenario1.retrieval import (  # noqa: E402
    canonical_plan_sha256,
    load_retrieval_plan,
    sha256_text,
)


DEFAULT_REGISTRY = (
    generator.INPUTS / "fellow_packages" / "aihc" / "domain_config.json"
)
PLANNER_SEED_TEXT = (
    " analysis of clinical validation evidence and reporting standards."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="Local files from the exact pinned Qwen model revision.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Defaults to retrieval.context_preflight_file from the selected "
            "registry."
        ),
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Run with the Modal-matched tokenizer dependency, for example: "
            "uv run --no-project --with 'transformers==5.8.0' "
            "python scripts/01_scenario_construction/"
            "03_preflight_retrieval_context.py ..."
        ) from exc

    registry = generator.load_registry(args.registry)
    plan_path = generator.INPUTS / registry["retrieval"]["plan_file"]
    plan = load_retrieval_plan(plan_path)
    tokenizer_dir = args.tokenizer_dir.resolve()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if _sha256_file(tokenizer_json) != plan["tokenizer"][
        "tokenizer_json_sha256"
    ]:
        raise ValueError("local tokenizer.json does not match the retrieval plan")
    config = json.loads((tokenizer_dir / "config.json").read_text())
    context_window = config.get("max_position_embeddings")
    if context_window != plan["budget"]["context_window_tokens"]:
        raise ValueError("local Qwen context window does not match the plan")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
    )
    target_planner_tokens = plan["budget"]["max_new_tokens"]
    planner_seed = PLANNER_SEED_TEXT * (target_planner_tokens + 1)
    planner_ids = tokenizer.encode(
        planner_seed,
        add_special_tokens=False,
    )[:target_planner_tokens]
    planner_message = tokenizer.decode(
        planner_ids,
        skip_special_tokens=True,
    )
    round_trip_count = len(
        tokenizer.encode(planner_message, add_special_tokens=False)
    )
    if round_trip_count != target_planner_tokens:
        raise ValueError("maximum-length planner fixture did not round-trip")

    cases = []
    preflight_registry = copy.deepcopy(registry)
    preflight_registry["retrieval"].pop("context_preflight_file", None)
    for treatment in ("clean", "injected"):
        record = generator.build_record(
            preflight_registry,
            "2-hop",
            treatment,
        )
        worker = next(
            event
            for event in record["trajectory_trace"]["full_events"]
            if event.get("agent_id") == "worker_1"
        )
        for thinking_mode, enable_thinking in (("off", False), ("on", True)):
            request = generator.build_generation_request(
                record,
                worker,
                thinking_mode=thinking_mode,
                upstream_message=planner_message,
            )
            rendered = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                enable_thinking=enable_thinking,
            )
            input_tokens = len(rendered["input_ids"])
            requested_total = (
                input_tokens
                + request["generation_settings"]["max_new_tokens"]
            )
            if requested_total > context_window:
                raise ValueError(
                    f"{treatment}/{thinking_mode} exceeds the context window"
                )
            cases.append({
                "treatment": treatment,
                "thinking_mode": thinking_mode,
                "input_token_count": input_tokens,
                "max_new_tokens": request["generation_settings"][
                    "max_new_tokens"
                ],
                "requested_sequence_tokens": requested_total,
                "headroom_tokens": context_window - requested_total,
            })

    result = {
        "schema_version": "spec_gap.context_preflight.v1",
        "generation_protocol_id": generator.generation_protocol_id_for_registry(
            registry
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_json_sha256": _sha256_file(tokenizer_json),
        "retrieval_plan_file": registry["retrieval"]["plan_file"],
        "retrieval_plan_sha256": canonical_plan_sha256(plan),
        "context_window_tokens": context_window,
        "planner_fixture": {
            "purpose": "maximum-length upstream planner-output preflight",
            "token_count": target_planner_tokens,
            "text_sha256": sha256_text(planner_message),
        },
        "cases": cases,
        "runtime_guard": (
            "src/infrastructure/modal_qwen_runner.py rejects any real rendered "
            "request whose input plus max_new_tokens exceeds the loaded model "
            "context; retrieved evidence is never silently truncated."
        ),
    }
    if args.out is not None:
        output_path = args.out.resolve()
    else:
        configured_output = registry["retrieval"].get(
            "context_preflight_file"
        )
        if not isinstance(configured_output, str) or not configured_output:
            raise ValueError(
                "set --out or registry retrieval.context_preflight_file"
            )
        output_path = (generator.INPUTS / configured_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote: {output_path}")
    for case in cases:
        print(
            f"{case['treatment']}/{case['thinking_mode']}: "
            f"{case['input_token_count']} input + {case['max_new_tokens']} "
            f"generated; {case['headroom_tokens']} tokens headroom"
        )


if __name__ == "__main__":
    main()
