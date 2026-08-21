"""Reproduce both Weeks 1-2 diagnostic baselines on runway artifacts.

This command does not run model inference. It reads the activation and response
files generated in Colab, then evaluates the Goldowsky-Dill-style logistic
probe and the repository's current LAT-style contrast-direction baseline under
the same stratified and leave-one-scenario-out splits.

Usage:
    python scripts/90_runway_reproduction/93_run_baselines.py
    python scripts/90_runway_reproduction/93_run_baselines.py --layers 16 20 24
    python scripts/90_runway_reproduction/93_run_baselines.py --artifact-root /path/to/artifacts
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "results" / "runway" / "week1_week2_baseline_comparison.json"
)

sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.runway_artifacts import (  # noqa: E402
    find_artifact_root,
    load_runway_artifacts,
)
from src.probes.lat_baseline import (  # noqa: E402
    evaluate_contrast_pair_lat_all_layers,
    evaluate_lat_all_layers,
    lat_results_to_dict,
)
from src.probes.linear_probe import evaluate_all_layers, results_to_dict  # noqa: E402


def _build_runway_pair_ids(artifact_root: Path) -> list[str]:
    """Pair the first/second colluder with first/second honest per scenario.

    NARCBench does not provide canonical contrast-pair IDs, so this pairing is
    deterministic but provisional. Phase 0 must use predeclared matched pairs.
    """

    response_path = (
        artifact_root / "02_collusion_probe" / "week2_collusion_probe_responses.json"
    )
    rows = json.loads(response_path.read_text())
    role_counts: dict[tuple[int, str], int] = {}
    pair_ids = []
    for row in rows:
        metadata = row["metadata"]
        scenario_idx = int(metadata["scenario_idx"])
        role = str(metadata["role"])
        key = (scenario_idx, role)
        within_role_index = role_counts.get(key, 0)
        role_counts[key] = within_role_index + 1
        pair_ids.append(f"scenario-{scenario_idx}-pair-{within_role_index}")
    return pair_ids


def run_baselines(
    artifact_root: Path | None = None,
    output_path: Path | None = None,
    layers: list[int] | None = None,
) -> dict:
    repo_root = PROJECT_ROOT
    artifact_root = artifact_root or find_artifact_root(repo_root)
    activations, labels, scenario_ids = load_runway_artifacts(artifact_root)
    contrast_pair_ids = _build_runway_pair_ids(artifact_root)

    if layers:
        missing = sorted(set(layers) - set(activations))
        if missing:
            raise ValueError(
                f"Requested layers are absent from the artifact: {missing}"
            )
        activations = {layer: activations[layer] for layer in layers}

    metadata = {"scenario_idx": scenario_ids}
    linear_stratified = evaluate_all_layers(
        activations,
        labels,
        n_splits=5,
        pca_components=50,
        random_state=42,
    )
    linear_lso = evaluate_all_layers(
        activations,
        labels,
        metadata=metadata,
        pca_components=50,
        leave_scenario_out=True,
        random_state=42,
    )
    lat_stratified = evaluate_lat_all_layers(
        activations,
        labels,
        n_splits=5,
        random_state=42,
    )
    lat_lso = evaluate_lat_all_layers(
        activations,
        labels,
        groups=scenario_ids,
        leave_group_out=True,
        random_state=42,
    )
    paired_lat_stratified = evaluate_contrast_pair_lat_all_layers(
        activations,
        labels,
        pair_ids=contrast_pair_ids,
        n_splits=5,
        random_state=42,
    )
    paired_lat_lso = evaluate_contrast_pair_lat_all_layers(
        activations,
        labels,
        pair_ids=contrast_pair_ids,
        groups=scenario_ids,
        leave_group_out=True,
        random_state=42,
    )

    output = {
        "experiment": "week1_week2_baseline_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "artifact_root": "external local or Colab artifact directory",
            "activations": "02_collusion_probe/week2_collusion_probe_activations.npz",
            "responses": "02_collusion_probe/week2_collusion_probe_responses.json",
        },
        "sample": {
            "n_prompts": int(len(labels)),
            "n_positive": int(labels.sum()),
            "n_negative": int((labels == 0).sum()),
            "n_scenarios": int(len(set(scenario_ids.tolist()))),
            "layers": sorted(activations),
        },
        "methods": {
            "goldowsky_dill_style": {
                "definition": "StandardScaler -> PCA(50) -> logistic regression",
                "stratified_cv": results_to_dict(linear_stratified),
                "leave_one_scenario_out": results_to_dict(linear_lso),
            },
            "lat_style": {
                "definition": "standardized class-centroid contrast direction with one-dimensional logistic calibration",
                "status": "development implementation; confirm against the locked contrast-pair LAT specification before final Phase 0 claims",
                "stratified_cv": lat_results_to_dict(lat_stratified),
                "leave_one_scenario_out": lat_results_to_dict(lat_lso),
            },
            "lat_contrast_pair": {
                "definition": "first PCA direction over symmetric matched positive-minus-negative activation differences, followed by one-dimensional logistic calibration",
                "pairing": "provisional NARCBench pairing: first/second colluder with first/second honest agent within each scenario",
                "status": "method implementation complete; runway pairing is provisional, and Phase 0 requires predeclared clean/injected contrast_pair_id values",
                "pair_preserving_cv": lat_results_to_dict(paired_lat_stratified),
                "leave_one_scenario_out": lat_results_to_dict(paired_lat_lso),
            },
        },
    }

    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    args = parser.parse_args()

    result = run_baselines(
        artifact_root=args.artifact_root,
        output_path=args.output,
        layers=args.layers,
    )
    print(json.dumps(result["sample"], indent=2))
