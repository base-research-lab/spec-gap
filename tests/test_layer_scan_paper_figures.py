import matplotlib.pyplot as plt

from src.analysis.layer_scan_paper_figures import save_paper_layer_scan_figures


def _paper_result():
    layers = {
        str(layer): {
            "auroc_mean": 0.5,
            "auroc_per_fold": [0.25, 0.75],
        }
        for layer in range(64)
    }
    strata = []
    controls = []
    for mode, checkpoints in (
        ("off", ("last_input_token", "last_visible_answer_token")),
        (
            "on",
            (
                "last_input_token",
                "last_reasoning_token",
                "last_visible_answer_token",
            ),
        ),
    ):
        for checkpoint in checkpoints:
            controls.append({
                "thinking_mode": mode,
                "checkpoint": checkpoint,
                "status": (
                    "failed_spurious_separation"
                    if checkpoint == "last_reasoning_token"
                    else "passed"
                ),
            })
            for agent_id, role, sample_count in (
                ("planner_1", "planner", 8),
                ("worker_1", "worker", 8),
                ("worker_2", "worker", 4),
                ("executor_1", "executor", 8),
            ):
                strata.append({
                    "status": "completed",
                    "thinking_mode": mode,
                    "agent_id": agent_id,
                    "agent_role": role,
                    "checkpoint": checkpoint,
                    "sample_count": sample_count,
                    "match_group_count": 2,
                    "layer_results": layers,
                })
    return {
        "claim_scope": "Exploratory construction-label signal only.",
        "strata": strata,
        "pre_injection_negative_control": {"checkpoint_controls": controls},
    }


def test_save_paper_figures_in_vector_and_preview_formats(tmp_path):
    paths = save_paper_layer_scan_figures(_paper_result(), tmp_path, dpi=72)

    assert len(paths) == 9
    assert {path.suffix for path in paths} == {".png", ".svg", ".pdf"}
    assert {path.stem for path in paths} == {
        "figure1_planner_negative_control",
        "figure2_shared_input_comparison",
        "appendix_full_layer_heatmap",
    }
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    assert plt.get_fignums() == []
