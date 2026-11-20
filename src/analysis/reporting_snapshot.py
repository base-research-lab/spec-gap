"""Compact, versioned inputs for rebuilding public Scenario 1 figures."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.analysis.depth_degradation import depth_result_analysis_tier
from src.infrastructure.modal_billing import ANALYSIS_TIERS, validate_analysis_tier
from src.infrastructure.model_profiles import ACTIVE_PROFILE


REPORTING_SNAPSHOT_SCHEMA = "spec_gap.scenario1_reporting_snapshot.v1"
REPORTING_ANALYSIS_TIERS = frozenset({*ANALYSIS_TIERS, "unclassified"})
REPORTING_LAYER = ACTIVE_PROFILE.reference_layer
REFERENCE_LAYERS = {REPORTING_LAYER, *ACTIVE_PROFILE.ablation_layers}
_LOW, _HIGH = ACTIVE_PROFILE.ablation_layers
PROBE_NAMES = {"goldowsky_dill_logistic", "lat_contrast_pair_pca"}
THINKING_MODES = {"off", "on"}


def build_reporting_snapshot(
    reference_result: dict[str, Any],
    all_layer_result: dict[str, Any],
    per_step_rows: Iterable[dict[str, Any]],
    *,
    analysis_revision: str,
    reporting_revision: str,
    source_artifacts: dict[str, dict[str, str]],
    model_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create the smallest data record needed to reproduce the public figures."""

    score_rows = [
        dict(row)
        for row in per_step_rows
        if int(row.get("layer", -1)) == REPORTING_LAYER
        and row.get("label_target") == "injection_present"
    ]
    if not score_rows:
        raise ValueError(
            f"No layer-{REPORTING_LAYER} construction-label score rows are available."
        )

    unique_trajectories: dict[str, dict[str, Any]] = {}
    for row in score_rows:
        unique_trajectories.setdefault(str(row["trajectory_id"]), row)
    outcome_counts = Counter(
        str(row["behavioral_outcome"]) for row in unique_trajectories.values()
    )

    compact_all_layer = {
        "schema_version": all_layer_result.get("schema_version"),
        "analysis_tier": depth_result_analysis_tier(all_layer_result),
        "claim_scope": all_layer_result["claim_scope"],
        "data_manifest_hash": all_layer_result.get("data_manifest_hash"),
        "experiment_id": all_layer_result["experiment_id"],
        "label_targets": list(all_layer_result["label_targets"]),
        "n_match_groups": int(all_layer_result["n_match_groups"]),
        "depth_metrics": [
            {
                "thinking_mode": row["thinking_mode"],
                "probe_name": row["probe_name"],
                "hop_mode": row["hop_mode"],
                "layer": int(row["layer"]),
                "auroc": row["auroc"],
                "path_mean_auroc": row["path_mean_auroc"],
            }
            for row in all_layer_result["depth_metrics"]
        ],
    }
    analysis_tier = reporting_sources_analysis_tier(
        model_configs,
        score_rows,
        (reference_result, all_layer_result),
    )
    snapshot = {
        "schema_version": REPORTING_SNAPSHOT_SCHEMA,
        "analysis_tier": analysis_tier,
        "analysis_revision": analysis_revision,
        "reporting_revision": reporting_revision,
        "source_artifacts": source_artifacts,
        "model_configs": model_configs,
        "sample": {
            "match_groups": int(reference_result["n_match_groups"]),
            "trajectory_runs": len(unique_trajectories),
            "agent_turns": len(
                {
                    (str(row["trajectory_id"]), int(row["hop_index"]))
                    for row in score_rows
                }
            ),
            "behavioral_outcomes": dict(sorted(outcome_counts.items())),
            "unsafe_action_executed_count": outcome_counts.get("executed", 0),
        },
        "reference_result": reference_result,
        "all_layer_result": compact_all_layer,
        "per_step_rows": score_rows,
    }
    validate_reporting_snapshot(snapshot)
    return snapshot


def validate_reporting_snapshot(snapshot: dict[str, Any]) -> None:
    """Fail closed when a public reporting snapshot is incomplete or inconsistent."""

    if snapshot.get("schema_version") != REPORTING_SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported Scenario 1 reporting snapshot schema.")
    for field in ("analysis_revision", "reporting_revision"):
        revision = snapshot.get(field)
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"Reporting snapshot requires {field}.")

    artifacts = snapshot.get("source_artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Reporting snapshot requires source artifact hashes.")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"Source artifact {name!r} must be an object.")
        path = artifact.get("path")
        checksum = artifact.get("sha256")
        if not isinstance(path, str) or not path:
            raise ValueError(f"Source artifact {name!r} has no path.")
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(f"Source artifact {name!r} has no SHA-256 checksum.")

    configs = snapshot.get("model_configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("Reporting snapshot requires model configuration metadata.")
    modes = {config.get("thinking_mode") for config in configs}
    if modes != THINKING_MODES:
        raise ValueError(
            "Model configurations must cover thinking off and thinking on."
        )
    decoding = [config.get("decoding_settings") for config in configs]
    if decoding[1:] != decoding[:-1]:
        raise ValueError("Thinking modes must use the same decoding settings.")

    reference = snapshot.get("reference_result")
    all_layer = snapshot.get("all_layer_result")
    rows = snapshot.get("per_step_rows")
    if not isinstance(reference, dict) or not isinstance(all_layer, dict):
        raise ValueError("Reporting snapshot requires reference and all-layer results.")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Reporting snapshot requires compact per-step scores.")
    source_tier = reporting_sources_analysis_tier(
        configs,
        rows,
        (reference, all_layer),
    )
    declared_tier = snapshot.get("analysis_tier", source_tier)
    if normalize_reporting_analysis_tier(declared_tier) != source_tier:
        raise ValueError(
            "Reporting snapshot analysis_tier does not match its model and score inputs."
        )
    if reference.get("label_targets") != ["injection_present"]:
        raise ValueError("Reference result must use the construction label.")
    if all_layer.get("label_targets") != ["injection_present"]:
        raise ValueError("All-layer result must use the construction label.")

    reference_layers = {int(row["layer"]) for row in reference["depth_metrics"]}
    if not REFERENCE_LAYERS.issubset(reference_layers):
        raise ValueError(
            f"Reference result must include layers {_LOW}, {REPORTING_LAYER}, and {_HIGH}."
        )
    all_layers = {int(row["layer"]) for row in all_layer["depth_metrics"]}
    if all_layers != set(range(ACTIVE_PROFILE.layer_count)):
        raise ValueError(
            "All-layer result must include layers 0 through "
            f"{ACTIVE_PROFILE.layer_count - 1}."
        )
    if {row.get("thinking_mode") for row in rows} != THINKING_MODES:
        raise ValueError("Per-step reporting rows must cover both thinking modes.")
    if {row.get("probe_name") for row in rows} != PROBE_NAMES:
        raise ValueError("Per-step reporting rows must cover both baseline probes.")
    if any(int(row.get("layer", -1)) != REPORTING_LAYER for row in rows):
        raise ValueError(
            f"Per-step reporting rows must contain only layer {REPORTING_LAYER}."
        )
    if any(row.get("label_target") != "injection_present" for row in rows):
        raise ValueError("Per-step reporting rows must use injection_present.")

    sample = snapshot.get("sample")
    if not isinstance(sample, dict):
        raise ValueError("Reporting snapshot requires a sample summary.")
    trajectory_count = len({str(row["trajectory_id"]) for row in rows})
    if int(sample.get("trajectory_runs", -1)) != trajectory_count:
        raise ValueError("Snapshot trajectory count does not match its per-step rows.")


def load_reporting_snapshot(path: str | Path) -> dict[str, Any]:
    """Load and validate one compact public reporting snapshot."""

    source = Path(path)
    try:
        snapshot = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load reporting snapshot {source}: {error}") from error
    if not isinstance(snapshot, dict):
        raise ValueError("Reporting snapshot must contain one JSON object.")
    validate_reporting_snapshot(snapshot)
    return snapshot


def write_reporting_snapshot(snapshot: dict[str, Any], path: str | Path) -> None:
    """Write a validated reporting snapshot as stable JSON."""

    validate_reporting_snapshot(snapshot)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local reporting input."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_reporting_analysis_tier(value: str) -> str:
    """Return one explicit tier used by reporting inputs."""

    if value == "unclassified":
        return value
    return validate_analysis_tier(value)


def reporting_sources_analysis_tier(
    model_configs: Iterable[dict[str, Any]],
    score_rows: Iterable[dict[str, Any]],
    depth_results: Iterable[dict[str, Any]],
) -> str:
    """Require every input to the reporting snapshot to identify one tier."""

    config_tiers = {
        normalize_reporting_analysis_tier(config.get("analysis_tier") or "unclassified")
        for config in model_configs
    }
    score_tiers = {
        normalize_reporting_analysis_tier(row.get("analysis_tier") or "unclassified")
        for row in score_rows
    }
    result_tiers = {depth_result_analysis_tier(result) for result in depth_results}
    if len(config_tiers) != 1 or len(score_tiers) != 1:
        raise ValueError("Reporting inputs must each contain one analysis tier.")
    if len(result_tiers) != 1:
        raise ValueError(
            "Reference and all-layer depth results must use one analysis tier."
        )
    if config_tiers != score_tiers or config_tiers != result_tiers:
        raise ValueError(
            "Reporting model provenance, probe scores, and depth results use "
            "different analysis tiers."
        )
    return next(iter(config_tiers))


def reporting_snapshot_analysis_tier(snapshot: dict[str, Any]) -> str:
    """Return the declared or backward-compatible inferred snapshot tier."""

    source_tier = reporting_sources_analysis_tier(
        snapshot.get("model_configs", []),
        snapshot.get("per_step_rows", []),
        (
            snapshot.get("reference_result", {}),
            snapshot.get("all_layer_result", {}),
        ),
    )
    declared_tier = snapshot.get("analysis_tier", source_tier)
    normalized = normalize_reporting_analysis_tier(declared_tier)
    if normalized != source_tier:
        raise ValueError(
            "Reporting snapshot analysis_tier does not match its model and score inputs."
        )
    return normalized
