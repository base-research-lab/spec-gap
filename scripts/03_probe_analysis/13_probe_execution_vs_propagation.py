#!/usr/bin/env python3
"""Executed-vs-propagated probe over lexical-sandbox executor activations (paper §5.6).

Reads the ``..._activations.npz`` / ``..._activations_manifest.json`` sidecars that
``09_run_paraphrase_dose.py`` writes next to each dose-response JSON, and reports a
group-safe logistic probe's AUROC for separating ``executed`` from ``propagated``
(Y != resisted), the executed-vs-propagated Cohen's d, and the grid-only vs grid+probe
incremental comparison.

Examples:

    # all captured runs under the default output dir, primary layer, LODO over domain
    python scripts/03_probe_analysis/13_probe_execution_vs_propagation.py

    # pick a layer and a specific run glob
    python scripts/03_probe_analysis/13_probe_execution_vs_propagation.py \
        --glob 'dose_response_*think_off*' --layer 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.execution_probe import load_probe_dataset, probe_executed_vs_propagated

DEFAULT_DIR = PROJECT_ROOT / "experiments/scenario1/outputs/paraphrase_sandbox"


def find_pairs(directory: Path, pattern: str) -> list[tuple[Path, Path]]:
    pairs = []
    for npz in sorted(directory.glob(pattern + "_activations.npz")):
        manifest = npz.with_name(npz.name.replace("_activations.npz", "_activations_manifest.json"))
        if manifest.is_file():
            pairs.append((npz, manifest))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="directory of sidecars")
    ap.add_argument("--glob", default="dose_response_*", help="run-name glob (no suffix)")
    ap.add_argument("--layer", type=int, default=None, help="captured layer to probe (default: primary)")
    ap.add_argument("--group-by", default="domain", help="group-safe CV grouping factor")
    ap.add_argument("--C", type=float, default=1.0, help="logistic inverse-regularization")
    args = ap.parse_args()

    directory = Path(args.dir)
    pairs = find_pairs(directory, args.glob)
    if not pairs:
        print(f"No activation sidecars match {args.glob!r} under {directory}.")
        print("Run 09_run_paraphrase_dose.py with capture_activations=True first.")
        return 1

    print(f"Pooling {len(pairs)} run(s):")
    for npz, _ in pairs:
        print(f"  {npz.name}")
    print()

    dataset = load_probe_dataset(pairs, layer=args.layer, group_by=args.group_by)
    print(
        f"Loaded {dataset.X.shape[0]} trajectories (Y != resisted) at layer "
        f"{dataset.layer}; excluded {dataset.n_excluded_resisted} resisted/clean.\n"
    )
    result = probe_executed_vs_propagated(dataset, C=args.C)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
