# Databricks notebook source
# DBTITLE 1,Header
# MAGIC %md
# MAGIC # 03 — Activation Extraction & Layer Probe
# MAGIC
# MAGIC Extracts residual-stream activations from Qwen3-32B at every layer for all
# MAGIC behavioral trajectories, then runs leave-one-domain-out linear probes to
# MAGIC identify which layers encode a clean-vs-injected construction signal.
# MAGIC
# MAGIC **Pipeline:** Uses existing `src/extraction` and `src/probes` modules:
# MAGIC 1. `trajectory.extract_trajectory_activations()` — TransformerLens extraction on Modal
# MAGIC 2. `saved_activations.build_activation_index()` — index saved `.pt` artifacts
# MAGIC 3. `layer_scan.run_construction_layer_scan()` — leave-one-domain-out probes across all layers

# COMMAND ----------

# DBTITLE 1,Setup: resolve repo root (portable)
import sys, os, json
from pathlib import Path
import numpy as np

# ── Repo root: portable across Databricks, local, and CI ──
if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    REPO_ROOT = Path(os.getcwd()).parent
else:
    REPO_ROOT = Path(".").resolve()
    _p = REPO_ROOT
    while _p != _p.parent:
        if (_p / "pyproject.toml").exists():
            REPO_ROOT = _p
            break
        _p = _p.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"Repo root: {REPO_ROOT}")

# COMMAND ----------

# DBTITLE 1,Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EDIT THIS CELL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from src.extraction.saved_activations import (
    CHECKPOINT_NAMES, LABEL_TARGETS,
    build_activation_index, load_probe_activation_batch,
    load_trajectory_records, trajectory_paths_for_analysis_tier,
)
from src.analysis.layer_scan import run_construction_layer_scan, layer_scan_table

# ── Paths ──
TRAJECTORY_ROOT = REPO_ROOT / "experiments/scenario1/outputs"
ARTIFACT_ROOT = REPO_ROOT  # saved_activations paths are relative to repo root

# ── Analysis tier (from qwen_modal runner) ──
ANALYSIS_TIER = "unclassified"  # or the tier tag from your run

# ── Probe target label ──
LABEL_TARGET = "injection_present"  # clean-vs-injected construction signal

print(f"Trajectory root: {TRAJECTORY_ROOT}")
print(f"Artifact root:   {ARTIFACT_ROOT}")
print(f"Analysis tier:   {ANALYSIS_TIER}")
print(f"Label target:    {LABEL_TARGET}")
print(f"Available checkpoints: {sorted(CHECKPOINT_NAMES)}")
print(f"Available label targets: {sorted(LABEL_TARGETS)}")

# COMMAND ----------

# DBTITLE 1,Load behavioral results
# ── Load trajectory records & build activation index ──
traj_paths = trajectory_paths_for_analysis_tier(TRAJECTORY_ROOT, ANALYSIS_TIER)
print(f"Found {len(traj_paths)} trajectory files for tier={ANALYSIS_TIER!r}")

records = load_trajectory_records(traj_paths)
print(f"Loaded {len(records)} trajectory records")

# ── Build activation index from saved .pt artifacts ──
index_rows = build_activation_index(
    records,
    analysis_tier=ANALYSIS_TIER,
    artifact_root=ARTIFACT_ROOT,
    require_local=True,
    verify_checksums=False,
)

print(f"\nActivation index: {len(index_rows)} rows")
print(f"  Checkpoints: {sorted(set(r['checkpoint'] for r in index_rows))}")
print(f"  Agents: {sorted(set(r['agent_id'] for r in index_rows))}")
print(f"  Match groups: {sorted(set(r['match_group_id'] for r in index_rows))}")
print(f"  Layers: {sorted(set(r['layer'] for r in index_rows[:100]))}")

# COMMAND ----------

# DBTITLE 1,Run probes per stratum
# ── Run layer-scan probes using src/analysis/layer_scan ──
# This calls evaluate_all_layers() from src/probes/linear_probe under the hood,
# with leave-one-match-group-out cross-validation.

scan_result = run_construction_layer_scan(
    index_rows,
    label_target=LABEL_TARGET,
    verify_checksums=False,
)

print(f"Layer scan complete.")
print(f"  Schema: {scan_result['schema_version']}")
print(f"  Probe: {scan_result['probe_name']}")
print(f"  Evaluation: {scan_result['evaluation_method']}")
print(f"  Strata completed: {scan_result['completed_strata']}")
print(f"  Strata skipped: {scan_result['skipped_strata']}")
print(f"  Layer selection allowed: {scan_result['layer_selection_allowed']}")
if scan_result['layer_selection_blockers']:
    print(f"  Blockers: {scan_result['layer_selection_blockers']}")

# COMMAND ----------

# DBTITLE 1,Results summary
# ── Flatten results into a summary table ──
import pandas as pd

table_rows = layer_scan_table(scan_result)
df = pd.DataFrame(table_rows)

if not df.empty:
    # Best layer per stratum
    best = df.loc[df.groupby(["thinking_mode", "agent_id", "checkpoint"])["auroc_mean"].idxmax()]
    print("\n★ Best layer per stratum:")
    print(best[["thinking_mode", "agent_id", "checkpoint", "layer",
                "auroc_mean", "auroc_std", "brier_mean", "ece_mean"]].to_string(index=False))

    # Save full results
    output_path = ARTIFACT_ROOT / "results/scenario1/layer_scan_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scan_result, f, indent=2)
    print(f"\nFull results saved: {output_path}")
else:
    print("⚠ No completed strata — check that activation artifacts are available.")

# COMMAND ----------

# DBTITLE 1,Negative control: planner activations
# ── Negative control: Planner (pre-retrieval) ──
# Planner never sees injected text, so its AUROC should be ~0.5.
# run_construction_layer_scan() already computes this via its
# pre_injection_negative_control field.

from src.analysis.layer_scan import NEGATIVE_CONTROL_AUROC_GUARDRAIL

print("="*60)
print("NEGATIVE CONTROL: Planner activations")
print("="*60)
print(f"Guardrail threshold: AUROC < {NEGATIVE_CONTROL_AUROC_GUARDRAIL}")
print()

neg_control = scan_result.get("pre_injection_negative_control", {})
status = neg_control.get("status", "not_available")

if status == "not_available":
    print("⚠ Planner negative control not available in this scan.")
elif neg_control.get("blocks_layer_selection"):
    print("✗ NEGATIVE CONTROL FAILED")
    print(f"  Status: {status}")
    print(f"  This blocks data-driven layer selection.")
    print(f"  Investigate structural differences in planner inputs.")
else:
    print("✓ Negative control passed.")
    print(f"  Status: {status}")
    planner_auroc = neg_control.get("best_auroc")
    if planner_auroc is not None:
        print(f"  Planner best AUROC: {planner_auroc:.3f} (below {NEGATIVE_CONTROL_AUROC_GUARDRAIL})")
    print("  The probe is detecting injection exposure, not structural confounds.")

# ── Limitations ──
print(f"\nLimitations:")
for lim in scan_result.get("limitations", []):
    print(f"  • {lim}")