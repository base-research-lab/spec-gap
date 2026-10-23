#!/usr/bin/env python3
"""Behavioral Scenario 1 matrix on a hosted OpenAI-compatible endpoint.

Default target: GLM-5.3 on Modal's shared endpoint (per-token billing, no GPU
to manage). Black-box only: outputs, tool calls and outcome labels, no
activations. Results live under outputs/trajectories/behavioral_endpoint/.

    # free: build the plan, print counts and a cost ceiling
    python scripts/02_model_execution/09_run_scenario1_endpoint.py \
        --domains aihc,convex,policy --thinking-modes off --action validate

    # paid: one trajectory first, then the cohort
    export MODAL_PROXY_TOKEN="wk-...ws-..."
    python scripts/02_model_execution/09_run_scenario1_endpoint.py \
        --domains aihc --thinking-modes off --action run \
        --max-new-trajectories 1 --max-cost-usd 2 \
        --confirm-paid-run RUN_ENDPOINT_BATCH
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.openai_endpoint import (  # noqa: E402
    EndpointClient,
    EndpointConfig,
)
from src.scenario1.batch import build_live_batch  # noqa: E402
from src.scenario1.behavioral_orchestrator import (  # noqa: E402
    load_turn,
    run_behavioral_trajectory,
    trajectory_output_path,
    write_trajectory,
    write_turn,
)
from src.scenario1.domain_grid import parse_domains, registry_paths_for_domains  # noqa: E402
from src.scenario1.generator import (  # noqa: E402
    ARTIFACT_ROOT,
    INPUTS,
    _agent_events,
    build_generation_request,
    load_registries,
)

CONFIRM = "RUN_ENDPOINT_BATCH"


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def _estimate(items: list[dict], config: EndpointConfig) -> dict:
    """Rough cost ceiling: chars/3.5 prompt tokens, every turn hits max_tokens."""

    prompt_tokens = 0
    turns = 0
    max_completion = 0
    for item in items:
        record = item["structural_record"]
        for event in _agent_events(record):
            request = build_generation_request(
                record, event, thinking_mode=item["thinking_mode"],
                upstream_message="x" * 6000,
                analysis_tier=item["analysis_tier"],
            )
            chars = sum(len(m.get("content") or "") for m in request["messages"])
            prompt_tokens += int(chars / 3.5)
            turns += 1
            max_completion += request["generation_settings"]["max_new_tokens"]
    price = config.pricing_per_mtok
    return {
        "model_turns": turns,
        "est_prompt_tokens": prompt_tokens,
        "max_completion_tokens": max_completion,
        "est_prompt_cost_usd": round(prompt_tokens * price["prompt"] / 1e6, 2),
        "ceiling_cost_usd": round(
            (prompt_tokens * price["prompt"] + max_completion * price["completion"])
            / 1e6, 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domains", default="")
    parser.add_argument("--registry-paths", default="")
    parser.add_argument("--thinking-modes", default="off")
    parser.add_argument("--analysis-tier", default="exploratory")
    parser.add_argument("--action", choices=("validate", "run"), default="validate")
    parser.add_argument("--confirm-paid-run", default="")
    parser.add_argument("--max-new-trajectories", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="hard stop once spend in this invocation reaches this")
    parser.add_argument("--model", default=None, help="default zai-org/GLM-5.3")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None,
                        choices=(None, "low", "high", "max"))
    parser.add_argument("--output-root", default="")
    args = parser.parse_args()

    config = EndpointConfig.from_env(
        model=args.model, base_url=args.base_url,
        reasoning_effort_on=args.reasoning_effort,
    )
    output_root = Path(args.output_root or (
        Path(ARTIFACT_ROOT) / "outputs/trajectories/behavioral_endpoint"
        / _slug(config.model)
    ))

    if args.domains and args.registry_paths:
        raise SystemExit("pass --domains or --registry-paths, not both")
    if args.domains:
        paths = [str(p) for p in registry_paths_for_domains(
            parse_domains(args.domains), inputs_root=INPUTS)]
    else:
        paths = [p.strip() for p in args.registry_paths.split(",") if p.strip()]
    if not paths:
        raise SystemExit("pass --domains or --registry-paths; no default cohort")
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"missing registry: {missing[0]}")

    modes = [m.strip() for m in args.thinking_modes.split(",") if m.strip()]
    plan = build_live_batch(
        load_registries(paths), thinking_modes=modes,
        output_root=output_root, analysis_tier=args.analysis_tier,
    )
    pending = [
        item for item in plan
        if not trajectory_output_path(
            output_root, item["analysis_tier"], item["thinking_mode"],
            item["trajectory_id"]).exists()
    ]
    selected = pending[: args.max_new_trajectories] if args.max_new_trajectories else pending
    preview = {
        "action": args.action,
        "model": config.model,
        "base_url": config.base_url,
        "output_root": str(output_root),
        "thinking_modes": modes,
        "analysis_tier": args.analysis_tier,
        "registries": len(paths),
        "total_trajectories": len(plan),
        "pending_trajectories": len(pending),
        "selected_trajectories": len(selected),
        "selected_estimate": _estimate(selected, config),
    }
    print(json.dumps(preview, indent=2), flush=True)
    if args.action == "validate":
        print("Validation passed. No endpoint was called.")
        return
    if args.confirm_paid_run != CONFIRM:
        raise SystemExit(f"paid run requires --confirm-paid-run {CONFIRM}")
    if args.max_cost_usd <= 0:
        raise SystemExit("paid run requires --max-cost-usd > 0")

    client = EndpointClient(config)

    def generate(request: dict) -> dict:
        cached = load_turn(request, output_root)
        if cached is not None:
            return cached
        if client.spent_usd >= args.max_cost_usd:
            raise SystemExit(
                f"cost ceiling reached: ${client.spent_usd:.4f} >= ${args.max_cost_usd}"
            )
        return client(request)

    outcomes: dict[str, int] = {}
    for index, item in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {item['trajectory_id']}", flush=True)
        record = run_behavioral_trajectory(
            item["structural_record"],
            thinking_mode=item["thinking_mode"],
            analysis_tier=item["analysis_tier"],
            generate_turn=generate,
            model_label=config.model,
            on_turn_complete=lambda req, res: write_turn(res, output_root, req),
        )
        path = write_trajectory(record, output_root, item["analysis_tier"])
        summary = record["behavioral_run_summary"]
        outcomes[summary["outcome_class"]] = outcomes.get(summary["outcome_class"], 0) + 1
        print(json.dumps({"path": str(path), **summary,
                          "spent_usd_this_run": round(client.spent_usd, 4)}),
              flush=True)
    print(json.dumps({"outcomes": outcomes,
                      "spent_usd_this_run": round(client.spent_usd, 4)}, indent=2))


if __name__ == "__main__":
    main()
