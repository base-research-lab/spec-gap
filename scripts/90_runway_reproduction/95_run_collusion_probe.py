"""
Week 2: Goldowsky-Dill linear probe on local development collusion data.

Trains probes on residual stream activations from Llama 3.1 8B Instruct
to distinguish colluding vs honest agents in committee deliberation scenarios.

This script uses the lightweight fixture in src/data/collusion_scenarios.py so
it can run without downloading upstream benchmark code. The canonical Week 2
run for reporting is notebooks/02_collusion_probe.ipynb, which pulls
NARCBench-Core upstream scenarios.

Usage:
    python scripts/90_runway_reproduction/95_run_collusion_probe.py
    python scripts/90_runway_reproduction/95_run_collusion_probe.py \
      --device cuda --pca 50 --batch-size 8
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.collusion_scenarios import build_prompts, get_scenarios
from src.extraction.residual_stream import (
    DEFAULT_LAYERS,
    extract_residual_stream,
    load_model,
)
from src.probes.linear_probe import evaluate_all_layers, results_to_dict


def _metadata_list_to_dict(metadata_list: list[dict]) -> dict:
    keys = metadata_list[0].keys()
    return {k: np.array([m[k] for m in metadata_list]) for k in keys}


def run_week2_probe(
    device: str = "cuda",
    pca_components: int | None = None,
    batch_size: int = 8,
):
    import huggingface_hub
    token = os.environ.get("HF_TOKEN_SANITY")
    if not token:
        raise EnvironmentError("Set HF_TOKEN_SANITY environment variable")
    huggingface_hub.login(token=token)

    scenarios = get_scenarios()
    prompts, labels, metadata_list = build_prompts(scenarios, chat_template=True)
    metadata_dict = _metadata_list_to_dict(metadata_list)

    print(f"Built {len(prompts)} prompts from {len(scenarios)} scenarios")
    print(f"  Colluder: {labels.sum()}, Honest: {(labels == 0).sum()}")

    print(f"\nLoading model on {device}...")
    model = load_model(device=device)

    print(f"Extracting activations at layers {DEFAULT_LAYERS} (batch_size={batch_size})...")
    activations = extract_residual_stream(model, prompts, batch_size=batch_size)

    del model
    torch.cuda.empty_cache()

    acts_np = {layer: acts.numpy().astype(np.float32) for layer, acts in activations.items()}
    for layer, arr in acts_np.items():
        print(f"  Layer {layer}: shape {arr.shape}")

    print(f"\n--- Stratified 5-fold CV (PCA={pca_components}) ---")
    results_cv = evaluate_all_layers(
        acts_np, labels, n_splits=5, pca_components=pca_components,
    )
    for layer, cv in results_cv.layer_results.items():
        print(f"  Layer {layer}: AUROC={cv.auroc_mean:.3f} +/- {cv.auroc_std:.3f}  "
              f"Brier={cv.brier_mean:.3f}  ECE={cv.ece_mean:.3f}")

    print(f"\n--- Leave-scenario-out CV (PCA={pca_components}) ---")
    results_lso = evaluate_all_layers(
        acts_np, labels, metadata=metadata_dict,
        leave_scenario_out=True, pca_components=pca_components,
    )
    for layer, cv in results_lso.layer_results.items():
        print(f"  Layer {layer}: AUROC={cv.auroc_mean:.3f} +/- {cv.auroc_std:.3f}  "
              f"Brier={cv.brier_mean:.3f}  ECE={cv.ece_mean:.3f}")

    output = {
        "experiment": "week2_collusion_probe",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "date": datetime.now().isoformat(),
        "n_scenarios": len(scenarios),
        "n_prompts": len(prompts),
        "n_colluder": int(labels.sum()),
        "n_honest": int((labels == 0).sum()),
        "token_position": "last",
        "pca_components": pca_components,
        "batch_size": batch_size,
        "stratified_cv": results_to_dict(results_cv),
        "leave_scenario_out_cv": results_to_dict(results_lso),
    }

    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "week2_collusion_probe.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pca", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    run_week2_probe(device=args.device, pca_components=args.pca, batch_size=args.batch_size)
