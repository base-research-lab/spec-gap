#!/usr/bin/env python3
"""Print or validate the canonical SPEC-GAP pipeline order.

Created: 2026-08-10.
Purpose: provide one checkout-relative entry point for discovering what to run
first, what each stage produces, and where paid or human gates begin.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

catalog = import_module("src.scenario1.pipeline_catalog")


def main() -> None:
    """Print the ordered catalog or check its repository bindings."""

    parser = argparse.ArgumentParser(
        description=(
            "Show the unique repository-wide S00-S21 run order. This command "
            "uses only the Python standard library and starts no remote work."
        )
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format for the catalog (default: text).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate stage order, entry-point paths, and runbook table drift.",
    )
    args = parser.parse_args()

    if args.check:
        catalog.validate_repository_pipeline(PROJECT_ROOT)

    if args.format == "json":
        print(json.dumps(catalog.pipeline_catalog_rows(), indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(catalog.render_markdown_table(), end="")
    else:
        print(catalog.render_text_catalog(), end="")

    if args.check:
        print("Pipeline catalog check passed.", file=sys.stderr)


if __name__ == "__main__":
    main()
