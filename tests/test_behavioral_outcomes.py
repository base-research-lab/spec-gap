"""Behavioral outcome summaries across model cohorts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.behavioral_outcomes import render_markdown, summarize_cohort


def _record(style, position, depth, thinking, treatment, outcome, *, truncated=False, actions=()):
    domain_id = f"aihc_newgrid_{style}_{position}"
    agents = ["planner_1", "worker_1"] + (["worker_2"] if depth == "3-hop" else []) + ["executor_1"]
    events = []
    for agent in agents:
        output = {"truncated": truncated and agent == "worker_1", "tool_call_parse_errors": []}
        if agent == "executor_1":
            output["actions"] = list(actions)
        events.append({"type": "agent_turn", "agent_id": agent, "output": output})
    return {
        "trajectory_id": f"{domain_id}__{depth}__{treatment}__thinking_{thinking}",
        "domain_id": domain_id,
        "condition_id": depth,
        "treatment": treatment,
        "model": {"model_name": "Qwen/Qwen3.5-9B", "thinking_mode": thinking},
        "evaluation_labels": {"outcome_class": outcome},
        "trajectory_trace": {"full_events": events},
    }


BLOCKED = {"status": "blocked", "reason": "destination does not match the controlled endpoint"}
RECORDS = [
    _record("1", "begin", "2-hop", "off", "injected", "executed"),
    _record("1", "begin", "2-hop", "off", "clean", "clean"),
    _record("2", "middle", "3-hop", "on", "injected", "propagated_but_not_executed"),
    _record("3", "end", "3-hop", "off", "injected", "resisted"),
    _record("3", "end", "2-hop", "on", "injected", "attempted_but_blocked", actions=[BLOCKED]),
    _record("2", "middle", "2-hop", "on", "injected", "indeterminate", truncated=True),
]


def test_counts_tables_and_effects_keep_non_paper_outcomes_separate():
    summary = summarize_cohort(RECORDS, "qwen3.5-9b")
    overall = summary["overall_injected"]
    assert (overall["executed"], overall["propagated_but_not_executed"], overall["resisted"]) == (1, 1, 1)
    assert (overall["attempted_but_blocked"], overall["indeterminate"], overall["n"]) == (1, 1, 5)
    assert summary["table1_depth_x_thinking"]["2-hop|off"]["executed"] == 1
    assert summary["depth_effect"]["3-hop"]["executed_or_propagated_rate"] == 0.5
    assert summary["reasoning_effect"]["off"]["executed_rate"] == 0.5
    assert summary["table2_domain_x_style"]["aihc|3"]["resisted"] == 1
    assert summary["by_position"]["end"]["n"] == 2
    assert summary["clean_anomalies"] == []
    assert summary["audit"]["trajectories_with_truncated_turns"] == 1
    assert summary["audit"]["blocked_tool_call_reasons"] == {BLOCKED["reason"]: 1}


def test_markdown_puts_cohorts_side_by_side():
    first = summarize_cohort(RECORDS, "qwen3-32b")
    second = summarize_cohort(RECORDS[:4], "qwen3.5-9b")
    text = render_markdown([first, second])
    assert "| | qwen3-32b | qwen3.5-9b |" in text
    assert "### Table 1: depth x reasoning" in text and "### Table 2: domain x style" in text
