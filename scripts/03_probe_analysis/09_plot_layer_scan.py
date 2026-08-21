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
from src.analysis.paper_inputs import (  # noqa: E402
    DEFAULT_PAPER_INPUT_POLICY,
    load_paper_input_policy,
    validate_embedded_paper_input_audit,
)
from src.extraction.saved_activations import (  # noqa: E402
    normalize_index_analysis_tier,
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
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--filename-prefix",
        default="scenario1_all_domains_",
        help=(
            "Paper-facing prefix for every output filename. Use a dated, "
            "self-identifying value for definitive analyses."
        ),
    )
    parser.add_argument(
        "--paper-input-policy",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_PAPER_INPUT_POLICY,
    )
    args = parser.parse_args()

    result = json.loads(args.input.read_text())
    analysis_tier = normalize_index_analysis_tier(result.get("analysis_tier"))
    paper_policy = load_paper_input_policy(args.paper_input_policy)
    paper_input_selection = validate_embedded_paper_input_audit(
        result,
        paper_policy,
    )
    paths = save_paper_layer_scan_figures(
        result,
        args.output_dir,
        dpi=args.dpi,
        filename_prefix=args.filename_prefix,
    )
    print(
        json.dumps(
            {
                "figure_count": len(paths),
                "figures": [path.as_posix() for path in paths],
                "claim_scope": result["claim_scope"],
                "analysis_tier": analysis_tier,
                "paper_input_policy_id": paper_input_selection["policy_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
