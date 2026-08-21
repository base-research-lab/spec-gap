"""Tests for the SPEC-GAP trajectory schema."""

import json

from src.pipeline.trajectory import TrajectoryLogger, now_utc


def test_optional_feedback_label_fields_serialize(tmp_path):
    logger = TrajectoryLogger(output_dir=tmp_path)
    timestamp = now_utc()

    record = logger.log_step(
        node_id="worker",
        role="worker",
        model="stub",
        timestamp_start=timestamp,
        timestamp_end=timestamp,
        input_context=[{"role": "user", "content": "summarize this document"}],
        output_message="summary",
        failure_mode="intent_free_compromise",
        trajectory_label="attempted_compromise",
        step_label="compromised_context",
        behavioral_sanity={"followed_injection": False, "preserved_task": True},
        label_notes="smoke test label",
        contrast_pair_id="scenario1-task01-worker-2hop",
    )

    payload = record.to_dict()
    assert payload["failure_mode"] == "intent_free_compromise"
    assert payload["trajectory_label"] == "attempted_compromise"
    assert payload["step_label"] == "compromised_context"
    assert payload["behavioral_sanity"]["preserved_task"] is True
    assert payload["label_notes"] == "smoke test label"
    assert payload["contrast_pair_id"] == "scenario1-task01-worker-2hop"

    written = json.loads(logger.path.read_text().strip())
    assert written == payload
    assert logger.validate() == []
