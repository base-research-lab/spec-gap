#!/usr/bin/env python3
"""Write a compact cost, token, truncation, and tool-use ledger for live runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.qwen_modal import (  # noqa: E402
    DEFAULT_GENERATION_PROTOCOL_ID,
)
from src.infrastructure.modal_billing import ANALYSIS_TIERS  # noqa: E402
from src.scenario1.validator import validate_payload  # noqa: E402


CSV_FIELDS = (
    "trajectory_id",
    "domain_id",
    "analysis_tier",
    "treatment",
    "condition_id",
    "thinking_mode",
    "outcome_class",
    "step_index",
    "agent_id",
    "input_tokens",
    "generated_tokens",
    "thinking_tokens",
    "final_output_tokens",
    "special_or_separator_tokens",
    "max_new_tokens",
    "finish_reason",
    "truncated",
    "thinking_complete",
    "elapsed_seconds",
    "estimated_h200_cost_usd",
    "modal_input_id",
    "modal_task_id",
    "tool_request_count",
    "controlled_endpoint_tool_request_count",
    "controlled_endpoint_in_generation",
)


def _generation_protocol_id(record: dict[str, Any]) -> str:
    return record.get(
        "generation_protocol_id",
        DEFAULT_GENERATION_PROTOCOL_ID,
    )


def _agent_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("trajectory_trace")
    if not isinstance(trace, dict):
        raise ValueError("trajectory_trace must be an object")
    events = trace.get("full_events")
    if not isinstance(events, list):
        raise ValueError("trajectory_trace.full_events must be an array")
    return [
        event
        for event in events
        if event.get("type") == "agent_turn"
    ]


def _sampling_reproducibility(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Audit saved trajectory and turn seeds without promising exact replay."""

    seeds: set[int] = set()
    turn_count = 0
    for record in records:
        trajectory_id = record.get("trajectory_id", "<unknown>")
        model = record.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"{trajectory_id}: model must be an object")
        seed = model.get("seed")
        decoding = model.get("decoding_settings")
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(decoding, dict)
            or decoding.get("seed") != seed
        ):
            raise ValueError(
                f"{trajectory_id}: model seed does not match decoding seed"
            )
        seeds.add(seed)
        for event in _agent_turns(record):
            turn_count += 1
            metadata = event.get("model_execution_metadata")
            settings = (
                metadata.get("generation_settings")
                if isinstance(metadata, dict) else None
            )
            if not isinstance(settings, dict) or settings.get("seed") != seed:
                raise ValueError(
                    f"{trajectory_id}: saved agent-turn seed does not match "
                    "the trajectory seed"
                )
    if turn_count == 0:
        raise ValueError("sampling audit requires at least one saved agent turn")
    return {
        "recorded_seeds": sorted(seeds),
        "model_turn_count_checked": turn_count,
        "trajectory_decoding_and_saved_turn_seeds_match": True,
        "bit_identical_stochastic_rerun_guaranteed": False,
        "analysis_unit": "saved_one_shot_sample",
        "limitation": (
            "A fixed seed does not guarantee identical stochastic CUDA output "
            "across runtime, library, hardware, or kernel changes."
        ),
    }


def _saved_analysis_tier(record: dict[str, Any]) -> str:
    tiers = set()
    for event in _agent_turns(record):
        metadata = event.get("model_execution_metadata")
        tiers.add(
            metadata.get("analysis_tier")
            if isinstance(metadata, dict) else None
        )
    if len(tiers) != 1:
        raise ValueError(
            f"{record.get('trajectory_id', '<unknown>')}: saved agent turns "
            "mix analysis tiers"
        )
    tier = next(iter(tiers), None)
    if tier is None:
        return "unclassified"
    if tier not in ANALYSIS_TIERS:
        raise ValueError(
            f"{record.get('trajectory_id', '<unknown>')}: invalid saved "
            "analysis tier"
        )
    return tier


def build_protocol_ledger(
    records: Iterable[dict[str, Any]],
    *,
    generation_protocol_id: str,
    modal_run_urls: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return per-turn rows and one aggregate summary for a single protocol."""

    selected = list(records)
    if not selected:
        raise ValueError("no trajectories matched the requested protocol and domain")
    if {
        _generation_protocol_id(record) for record in selected
    } != {generation_protocol_id}:
        raise ValueError("the ledger may contain only one generation protocol")
    analysis_tiers = {_saved_analysis_tier(record) for record in selected}
    if len(analysis_tiers) != 1:
        raise ValueError("the ledger may contain only one analysis tier")
    analysis_tier = next(iter(analysis_tiers))
    sampling_reproducibility = _sampling_reproducibility(selected)

    rows: list[dict[str, Any]] = []
    tool_request_details: list[dict[str, Any]] = []
    trajectory_summaries = []
    cost_by_treatment: defaultdict[str, float] = defaultdict(float)
    cost_by_thinking_mode: defaultdict[str, float] = defaultdict(float)
    outcomes: Counter[str] = Counter()
    total_tokens: Counter[str] = Counter()
    model_names = set()
    model_revisions = set()
    max_new_token_values = set()

    for record in selected:
        errors = validate_payload(record)
        if errors:
            raise ValueError(
                f"invalid trajectory {record.get('trajectory_id')}: "
                + "; ".join(errors)
            )
        model_names.add(record["model"]["model_name"])
        model_revisions.add(record["model"]["model_revision"])
        max_new_tokens = record["model"]["decoding_settings"]["max_new_tokens"]
        max_new_token_values.add(max_new_tokens)
        outcome = record["evaluation_labels"]["outcome_class"]
        outcomes[outcome] += 1
        endpoint = record["controlled_endpoint"]
        trajectory_cost = 0.0
        trajectory_truncated = False
        trajectory_tool_requests = 0
        trajectory_target_requests = 0

        for event in _agent_turns(record):
            output = event["output"]
            cost = event["cost_metadata"]
            usage = cost["token_usage"]
            calls = output.get("tool_call_requests", [])
            target_calls = [
                call
                for call in calls
                if call.get("arguments", {}).get("destination") == endpoint
            ]
            endpoint_in_generation = endpoint in output["raw_generated_text"]
            for call in calls:
                destination = call.get("arguments", {}).get("destination")
                tool_request_details.append({
                    "trajectory_id": record["trajectory_id"],
                    "treatment": record["treatment"],
                    "condition_id": record["condition_id"],
                    "thinking_mode": record["model"]["thinking_mode"],
                    "agent_id": event["agent_id"],
                    "tool_name": call.get("name"),
                    "destination": destination,
                    "targets_controlled_endpoint": destination == endpoint,
                })

            row = {
                "trajectory_id": record["trajectory_id"],
                "domain_id": record["domain_id"],
                "analysis_tier": analysis_tier,
                "treatment": record["treatment"],
                "condition_id": record["condition_id"],
                "thinking_mode": record["model"]["thinking_mode"],
                "outcome_class": outcome,
                "step_index": event["step_index"],
                "agent_id": event["agent_id"],
                "input_tokens": usage["input_tokens"],
                "generated_tokens": usage["generated_tokens"],
                "thinking_tokens": usage["thinking_tokens"],
                "final_output_tokens": usage["final_output_tokens"],
                "special_or_separator_tokens": usage[
                    "special_or_separator_tokens"
                ],
                "max_new_tokens": max_new_tokens,
                "finish_reason": output["finish_reason"],
                "truncated": output["truncated"],
                "thinking_complete": output["thinking_complete"],
                "elapsed_seconds": cost["elapsed_seconds"],
                "estimated_h200_cost_usd": cost["estimated_h200_cost_usd"],
                "modal_input_id": cost["modal_input_id"],
                "modal_task_id": cost["modal_task_id"],
                "tool_request_count": len(calls),
                "controlled_endpoint_tool_request_count": len(target_calls),
                "controlled_endpoint_in_generation": endpoint_in_generation,
            }
            rows.append(row)
            trajectory_cost += float(cost["estimated_h200_cost_usd"])
            trajectory_truncated = trajectory_truncated or bool(output["truncated"])
            trajectory_tool_requests += len(calls)
            trajectory_target_requests += len(target_calls)
            for field in (
                "input_tokens",
                "generated_tokens",
                "thinking_tokens",
                "final_output_tokens",
                "special_or_separator_tokens",
            ):
                total_tokens[field] += int(usage[field])

        cost_by_treatment[record["treatment"]] += trajectory_cost
        cost_by_thinking_mode[record["model"]["thinking_mode"]] += trajectory_cost
        trajectory_summaries.append({
            "trajectory_id": record["trajectory_id"],
            "treatment": record["treatment"],
            "condition_id": record["condition_id"],
            "thinking_mode": record["model"]["thinking_mode"],
            "outcome_class": outcome,
            "truncated": trajectory_truncated,
            "tool_request_count": trajectory_tool_requests,
            "controlled_endpoint_tool_request_count": trajectory_target_requests,
            "unsafe_action_executed": record["evaluation_labels"][
                "action_channel"
            ]["unsafe_action_executed"],
            "estimated_h200_cost_usd": round(trajectory_cost, 8),
        })

    max_row = max(rows, key=lambda row: int(row["generated_tokens"]))
    summary = {
        "schema_version": "spec_gap.generation_protocol_cost_summary.v2",
        "generation_protocol_id": generation_protocol_id,
        "analysis_tier": analysis_tier,
        "analysis_tier_uniform": True,
        "paper_cost_table_eligibility": "not_assessed_by_protocol_ledger",
        "modal_run_urls": list(modal_run_urls),
        "model_names": sorted(model_names),
        "model_revisions": sorted(model_revisions),
        "max_new_token_values": sorted(max_new_token_values),
        "sampling_reproducibility": sampling_reproducibility,
        "trajectory_count": len(selected),
        "model_turn_count": len(rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "truncated_turn_count": sum(bool(row["truncated"]) for row in rows),
        "truncated_trajectory_count": sum(
            item["truncated"] for item in trajectory_summaries
        ),
        "maximum_observed_turn": {
            "trajectory_id": max_row["trajectory_id"],
            "agent_id": max_row["agent_id"],
            "generated_tokens": max_row["generated_tokens"],
            "max_new_tokens": max_row["max_new_tokens"],
            "fraction_of_cap": round(
                max_row["generated_tokens"] / max_row["max_new_tokens"],
                6,
            ),
        },
        "token_totals": dict(total_tokens),
        "estimated_h200_cost_usd": {
            "by_treatment": {
                key: round(value, 8)
                for key, value in sorted(cost_by_treatment.items())
            },
            "by_thinking_mode": {
                key: round(value, 8)
                for key, value in sorted(cost_by_thinking_mode.items())
            },
            "total": round(
                sum(float(row["estimated_h200_cost_usd"]) for row in rows),
                8,
            ),
        },
        "tool_request_checks": {
            "all_request_count": len(tool_request_details),
            "controlled_endpoint_request_count": sum(
                item["targets_controlled_endpoint"]
                for item in tool_request_details
            ),
            "non_target_request_count": sum(
                not item["targets_controlled_endpoint"]
                for item in tool_request_details
            ),
            "controlled_endpoint_generation_mention_count": sum(
                bool(row["controlled_endpoint_in_generation"]) for row in rows
            ),
            "unsafe_action_trajectory_count": sum(
                item["unsafe_action_executed"] for item in trajectory_summaries
            ),
            "details": tool_request_details,
        },
        "trajectory_summaries": trajectory_summaries,
        "billing_note": (
            "Per-turn estimates cover measured model-turn time on one H200. "
            "They exclude CPU, memory, storage, network, credits, discounts, "
            "and some container overhead. Modal billing is authoritative."
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=PROJECT_ROOT / "experiments/scenario1/trajectories/live",
    )
    parser.add_argument("--generation-protocol-id", required=True)
    parser.add_argument(
        "--analysis-tier",
        required=True,
        choices=sorted((*ANALYSIS_TIERS, "unclassified")),
    )
    parser.add_argument("--domain-id")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--modal-run-url", action="append", default=[])
    parser.add_argument("--expected-trajectory-count", type=int)
    args = parser.parse_args()

    records = []
    for path in sorted(args.trajectory_root.rglob("*.json")):
        record = json.loads(path.read_text())
        if _generation_protocol_id(record) != args.generation_protocol_id:
            continue
        if _saved_analysis_tier(record) != args.analysis_tier:
            continue
        if args.domain_id is not None and record.get("domain_id") != args.domain_id:
            continue
        records.append(record)
    if (
        args.expected_trajectory_count is not None
        and len(records) != args.expected_trajectory_count
    ):
        raise ValueError(
            f"found {len(records)} trajectories, expected "
            f"{args.expected_trajectory_count}"
        )

    rows, summary = build_protocol_ledger(
        records,
        generation_protocol_id=args.generation_protocol_id,
        modal_run_urls=args.modal_run_url,
    )
    summary["declared_completeness"] = {
        "expected_trajectory_count": args.expected_trajectory_count,
        "observed_trajectory_count": len(records),
        "passed": (
            len(records) == args.expected_trajectory_count
            if args.expected_trajectory_count is not None else None
        ),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "trajectory_count": summary["trajectory_count"],
        "analysis_tier": summary["analysis_tier"],
        "declared_completeness": summary["declared_completeness"],
        "model_turn_count": summary["model_turn_count"],
        "truncated_turn_count": summary["truncated_turn_count"],
        "maximum_observed_turn": summary["maximum_observed_turn"],
        "estimated_h200_cost_usd": summary["estimated_h200_cost_usd"],
        "tool_request_checks": summary["tool_request_checks"],
        "output_csv": args.output_csv.as_posix(),
        "output_json": args.output_json.as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
