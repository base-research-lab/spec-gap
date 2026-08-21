"""Shared loaders for the historical runway activation artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def find_artifact_root(repo_root: Path) -> Path:
    """Find the external runway artifact directory."""

    candidates = []
    if os.environ.get("SPEC_GAP_ARTIFACT_ROOT"):
        candidates.append(Path(os.environ["SPEC_GAP_ARTIFACT_ROOT"]))
    candidates.extend(
        [
            repo_root / "artifacts",
            Path.home() / "Downloads" / "artifacts",
            Path("/content/drive/MyDrive/spec-gap-activation-probe/artifacts"),
        ]
    )
    expected = Path("02_collusion_probe/week2_collusion_probe_activations.npz")
    for candidate in candidates:
        if (candidate / expected).exists():
            return candidate
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find runway activation artifact. Searched:\n" + searched
    )


def load_runway_artifacts(
    artifact_root: Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Load activation arrays, labels, and scenario IDs."""

    collusion_dir = artifact_root / "02_collusion_probe"
    activation_path = collusion_dir / "week2_collusion_probe_activations.npz"
    response_path = collusion_dir / "week2_collusion_probe_responses.json"

    npz = np.load(activation_path)
    activations = {
        int(name.removeprefix("layer_")): npz[name]
        for name in npz.files
        if name.startswith("layer_")
    }
    labels = np.asarray(npz["labels"], dtype=int)

    responses = json.loads(response_path.read_text())
    scenario_ids = np.asarray(
        [row["metadata"]["scenario_idx"] for row in responses], dtype=int
    )
    if len(labels) != len(scenario_ids):
        raise ValueError(
            "Activation labels and response metadata have different lengths."
        )
    return activations, labels, scenario_ids
