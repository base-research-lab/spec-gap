"""
Sanity check: replicate Marks & Tegmark (2023) true/false linear separability.

Runs difference-in-means on residual stream activations from Llama 3 8B Instruct.
If layer 20 shows clean separation (AUROC > 0.8), the extraction pipeline works.

Usage:
    python scripts/90_runway_reproduction/91_sanity_check_geometry.py
    python scripts/90_runway_reproduction/91_sanity_check_geometry.py --device cpu --n-statements 20
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.extraction.residual_stream import (
    DEFAULT_LAYERS,
    extract_residual_stream,
    load_model,
)

TRUE_STATEMENTS = [
    "The Earth orbits the Sun.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The speed of light is approximately 300,000 kilometers per second.",
    "Humans have 23 pairs of chromosomes.",
    "The chemical formula for water is H2O.",
    "The Great Wall of China is visible from space with aid.",
    "DNA stands for deoxyribonucleic acid.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Diamonds are made of carbon.",
    "The Moon causes tides on Earth.",
    "Shakespeare wrote Hamlet.",
    "Oxygen is necessary for human respiration.",
    "The Nile is the longest river in Africa.",
    "Iron is attracted to magnets.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The human heart has four chambers.",
    "Jupiter is the largest planet in our solar system.",
    "Sound travels faster in water than in air.",
    "The boiling point of nitrogen is below zero degrees Celsius.",
    "Antibiotics are used to treat bacterial infections.",
    "The Amazon rainforest produces a significant portion of the world's oxygen.",
    "Gold is a chemical element with the symbol Au.",
    "The Sahara is the largest hot desert in the world.",
    "Insulin regulates blood sugar levels.",
    "Light travels faster than sound.",
    "The human body contains more bacteria than human cells.",
    "Mount Everest is the tallest mountain above sea level.",
    "Electrons have a negative charge.",
    "The liver is the largest internal organ in the human body.",
    "Gravity pulls objects toward the center of the Earth.",
]

FALSE_STATEMENTS = [
    "The Earth is the largest planet in the solar system.",
    "Water freezes at 50 degrees Celsius.",
    "The speed of light is approximately 300 kilometers per second.",
    "Humans have 46 pairs of chromosomes.",
    "The chemical formula for water is CO2.",
    "The Great Wall of China was built in the 20th century.",
    "DNA stands for dioxyribonucleic acid.",
    "The Atlantic Ocean is the largest ocean on Earth.",
    "Diamonds are made of iron.",
    "The Sun causes tides on Earth.",
    "Shakespeare wrote The Odyssey.",
    "Nitrogen is necessary for human respiration.",
    "The Amazon is the longest river in Africa.",
    "Wood is attracted to magnets.",
    "Photosynthesis converts moonlight into chemical energy.",
    "The human heart has two chambers.",
    "Saturn is the largest planet in our solar system.",
    "Sound travels faster in a vacuum than in air.",
    "The boiling point of nitrogen is above 100 degrees Celsius.",
    "Antibiotics are used to treat viral infections.",
    "The Sahara desert produces a significant portion of the world's oxygen.",
    "Gold is a chemical element with the symbol Fe.",
    "Antarctica is the largest hot desert in the world.",
    "Adrenaline regulates blood sugar levels.",
    "Sound travels faster than light.",
    "The human body contains no bacteria.",
    "Mount Kilimanjaro is the tallest mountain above sea level.",
    "Electrons have a positive charge.",
    "The spleen is the largest internal organ in the human body.",
    "Gravity pushes objects away from the center of the Earth.",
]


def run_sanity_check(device: str = "cuda", n_statements: int | None = None):
    import huggingface_hub
    token = os.environ.get("HF_TOKEN_SANITY")
    if not token:
        raise EnvironmentError("Set HF_TOKEN_SANITY environment variable")
    huggingface_hub.login(token=token)

    true_stmts = TRUE_STATEMENTS[:n_statements] if n_statements else TRUE_STATEMENTS
    false_stmts = FALSE_STATEMENTS[:n_statements] if n_statements else FALSE_STATEMENTS

    assert len(true_stmts) == len(false_stmts), "Balanced classes required"

    all_prompts = true_stmts + false_stmts
    labels = np.array([1] * len(true_stmts) + [0] * len(false_stmts))

    print(f"Loading model on {device}...")
    model = load_model(device=device)

    print(f"Extracting activations for {len(all_prompts)} statements at layers {DEFAULT_LAYERS}...")
    activations = extract_residual_stream(model, all_prompts, batch_size=1)

    del model
    torch.cuda.empty_cache()

    results = {}
    for layer, acts in activations.items():
        X = acts.numpy().astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels,
        )

        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X_train, y_train)

        y_prob = probe.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        accuracy = probe.score(X_test, y_test)

        mean_true = X[labels == 1].mean(axis=0)
        mean_false = X[labels == 0].mean(axis=0)
        diff = mean_true - mean_false
        cosine_norm = np.linalg.norm(diff)

        results[layer] = {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
            "diff_means_norm": float(cosine_norm),
            "n_train": len(y_train),
            "n_test": len(y_test),
        }

        status = "PASS" if auroc > 0.8 else "CHECK"
        print(f"  Layer {layer}: AUROC={auroc:.3f}  Acc={accuracy:.3f}  [{status}]")

    output = {
        "experiment": "sanity_check_geometry_of_truth",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "date": datetime.now().isoformat(),
        "n_true": len(true_stmts),
        "n_false": len(false_stmts),
        "token_position": "last",
        "layers": results,
    }

    out_dir = Path(__file__).resolve().parents[2] / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sanity_check_geometry_of_truth.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    layer_20 = results.get(20, {})
    if layer_20.get("auroc", 0) > 0.8:
        print("\nSanity check PASSED: layer 20 AUROC > 0.8, extraction pipeline is producing sensible activations.")
    else:
        print(f"\nSanity check NEEDS REVIEW: layer 20 AUROC = {layer_20.get('auroc', 'N/A')}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-statements", type=int, default=None)
    args = parser.parse_args()

    run_sanity_check(device=args.device, n_statements=args.n_statements)
