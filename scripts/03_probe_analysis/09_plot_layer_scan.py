#!/usr/bin/env python3
"""Create guarded exploratory figures from a saved layer-scan result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.layer_scan_paper_figures import (  # noqa: E402
    save_paper_layer_scan_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a saved layer scan without model or GPU compute."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/construction_layer_scan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/scenario1/figures/paper",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    result = json.loads(args.input.read_text())
    paths = save_paper_layer_scan_figures(result, args.output_dir, dpi=args.dpi)
    print(json.dumps({
        "figure_count": len(paths),
        "figures": [path.as_posix() for path in paths],
        "claim_scope": result["claim_scope"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
