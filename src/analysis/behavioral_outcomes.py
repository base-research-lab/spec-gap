"""Behavioral outcome tables for Scenario 1 cohorts from any model profile.

Reads completed live trajectory records (for example Qwen3-32B and Qwen3.5-9B)
and reports the paper's behavioral comparisons:
outcomes by delegation depth and reasoning mode (Table 1), by domain and
injection style (Table 2), and by insertion position, plus the depth,
reasoning, and explicitness effects stated in Sections 5.2-5.3. Outcome
classes outside the paper's three (``attempted_but_blocked``,
``indeterminate``) are counted separately rather than folded in.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import fisher_exact


PAPER_OUTCOMES = ("executed", "propagated_but_not_executed", "resisted")
OTHER_OUTCOMES = ("attempted_but_blocked", "indeterminate")
STYLE_LABELS = {"1": "calibration_line", "2": "self_replicating", "3": "tool_binding"}
_CELL = re.compile(r"^(?P<domain>.+)_newgrid_(?P<style>\d+)_(?P<position>begin|middle|end)$")


def load_records(root: str | Path) -> list[dict[str, Any]]:
    """Load every ``<mode>/*.json`` trajectory under one cohort root."""

    paths = sorted(Path(root).glob("*/*.json"))
    if not paths:
        raise ValueError(f"no trajectory records under {root}")
    return [json.loads(path.read_text()) for path in paths]


def _row(record: dict[str, Any]) -> dict[str, Any]:
    match = _CELL.match(record["domain_id"])
    if match is None:
        raise ValueError(f"{record['trajectory_id']}: domain_id is not a grid cell")
    turns = [event for event in record["trajectory_trace"]["full_events"]
             if event.get("type") == "agent_turn"]
    executor = next(event for event in turns if event["agent_id"] == "executor_1")
    return {
        "trajectory_id": record["trajectory_id"],
        "domain": match["domain"],
        "style": match["style"],
        "position": match["position"],
        "depth": record["condition_id"],
        "thinking": record["model"]["thinking_mode"],
        "treatment": record["treatment"],
        "outcome": record["evaluation_labels"]["outcome_class"],
        "model": record["model"].get("model_name"),
        "truncated_turns": sum(bool(event["output"].get("truncated")) for event in turns),
        "executor_actions": executor["output"].get("actions") or [],
        "executor_parse_errors": executor["output"].get("tool_call_parse_errors") or [],
    }


def _counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row["outcome"] for row in rows)
    result = {name: counter.get(name, 0) for name in PAPER_OUTCOMES + OTHER_OUTCOMES}
    result["n"] = sum(counter.values())
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _contrast(rows: list[dict[str, Any]], key: str, levels: tuple[str, str]) -> dict[str, Any]:
    """Execution and carried-downstream rates for two levels, with Fisher tests."""

    result: dict[str, Any] = {}
    tables = {"executed": [], "executed_or_propagated": [], "resisted": []}
    for level in levels:
        subset = [row for row in rows if row[key] == level]
        counts = _counts(subset)
        n = counts["n"]
        carried = counts["executed"] + counts["propagated_but_not_executed"]
        result[level] = {
            **counts,
            "executed_rate": _rate(counts["executed"], n),
            "executed_or_propagated_rate": _rate(carried, n),
            "resisted_rate": _rate(counts["resisted"], n),
        }
        tables["executed"].append([counts["executed"], n - counts["executed"]])
        tables["executed_or_propagated"].append([carried, n - carried])
        tables["resisted"].append([counts["resisted"], n - counts["resisted"]])
    result["fisher_exact_p"] = {
        name: (round(float(fisher_exact(table)[1]), 6)
               if all(sum(row) for row in table) else None)
        for name, table in tables.items()
    }
    return result


def summarize_cohort(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    rows = [_row(record) for record in records]
    injected = [row for row in rows if row["treatment"] == "injected"]
    clean = [row for row in rows if row["treatment"] == "clean"]

    table1 = {
        f"{depth}|{thinking}": _counts(
            row for row in injected if row["depth"] == depth and row["thinking"] == thinking
        )
        for depth in ("2-hop", "3-hop") for thinking in ("off", "on")
    }
    domains = sorted({row["domain"] for row in injected})
    styles = sorted({row["style"] for row in injected}, key=int)
    table2 = {
        f"{domain}|{style}": _counts(
            row for row in injected if row["domain"] == domain and row["style"] == style
        )
        for domain in domains for style in styles
    }
    by_style = {style: _counts(row for row in injected if row["style"] == style) for style in styles}
    by_position = {
        position: _counts(row for row in injected if row["position"] == position)
        for position in ("begin", "middle", "end")
    }
    executed_by_style = [by_style[style]["executed"] for style in styles]
    resisted_by_style = [by_style[style]["resisted"] for style in styles]

    blocked_reasons = Counter(
        action.get("reason")
        for row in injected for action in row["executor_actions"]
        if action.get("status") == "blocked"
    )
    return {
        "label": label,
        "models": sorted({str(row["model"]) for row in rows}),
        "trajectories": len(rows),
        "injected": len(injected),
        "clean": len(clean),
        "overall_injected": _counts(injected),
        "table1_depth_x_thinking": table1,
        "depth_effect": _contrast(injected, "depth", ("2-hop", "3-hop")),
        "reasoning_effect": _contrast(injected, "thinking", ("off", "on")),
        "explicitness": {
            "by_style": by_style,
            "style_labels": {style: STYLE_LABELS.get(style, style) for style in styles},
            "executed_monotonic_increasing": executed_by_style == sorted(executed_by_style),
            "resisted_monotonic_increasing": resisted_by_style == sorted(resisted_by_style),
        },
        "table2_domain_x_style": table2,
        "by_position": by_position,
        "clean_anomalies": sorted(
            row["trajectory_id"] for row in clean if row["outcome"] != "clean"
        ),
        "audit": {
            "trajectories_with_truncated_turns": sum(bool(row["truncated_turns"]) for row in rows),
            "blocked_tool_call_reasons": dict(blocked_reasons),
            "executor_parse_error_trajectories": sum(bool(row["executor_parse_errors"]) for row in rows),
        },
    }


def _fmt_counts(counts: dict[str, int]) -> str:
    extra = counts["attempted_but_blocked"] + counts["indeterminate"]
    text = (f"{counts['executed']} / {counts['propagated_but_not_executed']} / "
            f"{counts['resisted']}")
    return text + (f" (+{extra} other)" if extra else "")


def render_markdown(summaries: list[dict[str, Any]]) -> str:
    """Render side-by-side cohort tables; cells read executed / propagated / resisted."""

    labels = [summary["label"] for summary in summaries]
    header = "| | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels) + "\n"

    def table(title: str, keys: list[str], getter) -> str:
        lines = [f"### {title}\n", header]
        for key in keys:
            lines.append(f"| {key} | " + " | ".join(getter(summary, key) for summary in summaries) + " |\n")
        return "".join(lines) + "\n"

    out = ["# Scenario 1 behavioral outcomes\n\n",
           "Cells are executed / propagated / resisted among injected trajectories; "
           "`other` counts attempted_but_blocked and indeterminate.\n\n"]
    out.append(table("Overall", ["injected"], lambda s, _k: _fmt_counts(s["overall_injected"])))
    keys1 = list(summaries[0]["table1_depth_x_thinking"])
    out.append(table("Table 1: depth x reasoning", keys1,
                     lambda s, k: _fmt_counts(s["table1_depth_x_thinking"].get(k, _counts([])))))

    def effect(name: str, levels: tuple[str, str]) -> str:
        def cell(summary: dict[str, Any], metric: str) -> str:
            data = summary[name]
            rates = " vs ".join(
                f"{data[level][metric + '_rate']}" for level in levels
            )
            return f"{rates} (p={data['fisher_exact_p'][metric]})"
        return table(f"{name.replace('_', ' ').title()} ({levels[0]} vs {levels[1]})",
                     ["executed", "executed_or_propagated", "resisted"], cell)

    out.append(effect("depth_effect", ("2-hop", "3-hop")))
    out.append(effect("reasoning_effect", ("off", "on")))
    styles = list(summaries[0]["explicitness"]["by_style"])
    out.append(table("Explicitness: by injection style", styles,
                     lambda s, k: _fmt_counts(s["explicitness"]["by_style"].get(k, _counts([])))))
    out.append(table("Explicitness: monotonic trends", ["executed rises", "resisted rises"],
                     lambda s, k: str(s["explicitness"][
                         "executed_monotonic_increasing" if k.startswith("executed")
                         else "resisted_monotonic_increasing"])))
    keys2 = list(summaries[0]["table2_domain_x_style"])
    out.append(table("Table 2: domain x style", keys2,
                     lambda s, k: _fmt_counts(s["table2_domain_x_style"].get(k, _counts([])))))
    out.append(table("By insertion position", ["begin", "middle", "end"],
                     lambda s, k: _fmt_counts(s["by_position"][k])))
    out.append(table("Audit", ["clean anomalies", "truncated trajectories",
                               "executor parse errors", "blocked reasons"],
                     lambda s, k: {
                         "clean anomalies": str(len(s["clean_anomalies"])),
                         "truncated trajectories": str(s["audit"]["trajectories_with_truncated_turns"]),
                         "executor parse errors": str(s["audit"]["executor_parse_error_trajectories"]),
                         "blocked reasons": json.dumps(s["audit"]["blocked_tool_call_reasons"]),
                     }[k]))
    return "".join(out)
