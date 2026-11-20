#!/usr/bin/env python3
"""Compare Scenario 1 behavioral outcomes across cohorts (models or tiers).

Each ``--cohort LABEL=PATH`` points at one tier folder holding ``<mode>/*.json``
records, for example the Qwen3-32B and Qwen3.5-9B live tiers:

    python scripts/04_reporting/19_summarize_behavioral_outcomes.py \
        --cohort qwen3-32b=experiments/scenario1/trajectories/live/definitive \
        --cohort qwen3.5-9b=experiments/scenario1/trajectories/qwen3_5_9b/live/definitive \
        --output-json results/scenario1/behavioral_comparison.json \
        --output-md results/scenario1/behavioral_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.behavioral_outcomes import (  # noqa: E402
    load_records,
    render_markdown,
    summarize_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--cohort", action="append", required=True,
                        help="LABEL=PATH to a tier folder of <mode>/*.json records")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summaries = []
    for item in args.cohort:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--cohort must be LABEL=PATH, got {item!r}")
        summaries.append(summarize_cohort(load_records(path), label))
    markdown = render_markdown(summaries)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
