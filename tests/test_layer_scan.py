import hashlib

import pytest
import torch

from src.analysis.layer_scan import layer_scan_table, run_construction_layer_scan


def _index_rows(tmp_path):
    rows = []
    for group_index, group in enumerate(("group-a", "group-b")):
        for label in (0, 1):
            trajectory_id = f"{group}-{label}"
            path = tmp_path / f"{trajectory_id}.pt"
            base = torch.tensor([
                [float(label), float(group_index), 0.0],
                [float(label * 3), float(group_index), 1.0],
            ])
            torch.save({
                "activations": base,
                "layers": [0, 1],
                "artifact_format_version": "spec_gap.activation_positions.v1",
                "primary_checkpoint": "last_input_token",
                "position_activations": {"last_input_token": base},
            }, path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "trajectory_id": trajectory_id,
                "step_index": 2,
                "agent_id": "worker_1",
                "agent_role": "worker",
                "thinking_mode": "off",
                "checkpoint": "last_input_token",
                "match_group_id": group,
                "matched_pair_id": f"{group}__2hop",
                "delegation_depth": "2-hop",
                "treatment": "injected" if label else "clean",
                "labels": {
                    "injection_present": label,
                    "unsafe_action_executed": 0,
                    "output_adoption": 0,
                },
                "local_path": path.as_posix(),
                "checksum_sha256": checksum,
                "rendered_prompt_hash": "same-prompt",
            })
    return rows


def test_layer_scan_is_group_safe_and_returns_each_layer(tmp_path):
    result = run_construction_layer_scan(_index_rows(tmp_path))

    assert result["evaluation_method"] == "leave_one_match_group_out"
    assert result["selection_status"] == "exploratory_small_n"
    assert result["layer_selection_allowed"] is False
    assert result["pre_injection_negative_control"]["status"] == "not_available"
    assert result["completed_strata"] == 1
    stratum = result["strata"][0]
    assert stratum["match_group_count"] == 2
    assert stratum["class_counts"] == {"negative": 2, "positive": 2, "missing": 0}
    assert set(stratum["layer_results"]) == {"0", "1"}
    assert len(stratum["layer_results"]["0"]["auroc_per_fold"]) == 2
    table = layer_scan_table(result)
    assert len(table) == 2
    assert {row["layer"] for row in table} == {0, 1}


def test_layer_scan_skips_one_class_behavioral_target(tmp_path):
    result = run_construction_layer_scan(
        _index_rows(tmp_path),
        label_target="unsafe_action_executed",
    )

    assert result["completed_strata"] == 0
    assert result["skipped_strata"] == 1
    assert "only one class" in result["strata"][0]["skip_reason"]


def test_layer_scan_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError, match="label_target must be one of"):
        run_construction_layer_scan(_index_rows(tmp_path), label_target="latent")


def test_planner_false_signal_blocks_layer_selection(tmp_path):
    rows = _index_rows(tmp_path)
    for row in rows:
        row["agent_id"] = "planner_1"
        row["agent_role"] = "planner"

    result = run_construction_layer_scan(rows)

    control = result["pre_injection_negative_control"]
    assert control["matched_prompt_pair_count"] == 2
    assert control["prompt_hash_mismatches"] == []
    assert control["status"] == "failed_spurious_separation"
    assert len(control["checkpoint_controls"]) == 1
    assert control["checkpoint_controls"][0]["checkpoint"] == "last_input_token"
    assert result["strata"][0]["negative_control_status"] == (
        "failed_spurious_separation"
    )
    assert result["strata"][0]["exploratory_ranking_qualified"] is False
    assert result["layer_selection_allowed"] is False


def test_layer_scan_uses_strict_paired_control_audit(tmp_path):
    rows = _index_rows(tmp_path)
    for row in rows:
        row["agent_id"] = "planner_1"
        row["agent_role"] = "planner"
    audit = {
        "strict_planner_input_control": {
            "status": "passed",
            "reason": "Planner inputs match.",
            "mode_controls": [{"thinking_mode": "off", "status": "passed"}],
        },
        "stochastic_planner_output_control": {
            "status": "not_available",
            "blocks_stratum_ranking_until_calibrated": False,
        },
    }

    result = run_construction_layer_scan(rows, control_audit=audit)

    control = result["pre_injection_negative_control"]
    assert control["status"] == "strict_input_passed"
    assert control["blocks_layer_selection"] is False
    assert control["qualified_mode_checkpoints"] == [{
        "thinking_mode": "off",
        "checkpoint": "last_input_token",
    }]
    assert result["strata"][0]["exploratory_ranking_qualified"] is True
