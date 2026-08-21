import re
import runpy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.infrastructure.modal_billing import (
    aggregate_billing_rows,
    as_decimal,
    scenario1_billing_tags,
    validate_analysis_tier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SUMMARY_SCRIPT = (
    PROJECT_ROOT
    / "scripts/02_model_execution/07_summarize_scenario1_protocol.py"
)
BILLING_RECONCILIATION_SCRIPT = (
    PROJECT_ROOT
    / "scripts/02_model_execution/08_reconcile_modal_billing.py"
)


def test_scenario1_billing_tags_are_stable_and_specific():
    assert scenario1_billing_tags(
        domain_ids=["macro", "aihc", "macro"],
        generation_protocol_ids=["controlled_v2_5000"],
        run_kind="batch",
        analysis_tier="definitive",
    ) == {
        "project": "spec-gap",
        "component": "qwen3-inference",
        "scenario": "scenario1",
        "domains": "aihc.macro",
        "generation_protocols": "controlled_v2_5000",
        "run_kind": "batch",
        "analysis_tier": "definitive",
    }


def test_scenario1_billing_tags_fit_modal_limits_for_nine_domains():
    tags = scenario1_billing_tags(
        domain_ids=[
            "aihc_newgrid_12",
            "convex_newgrid_28",
            "fin_newgrid_20",
            "kg_newgrid_12",
            "macro_newgrid_28",
            "neuro_newgrid_20",
            "petro_newgrid_12",
            "policy_newgrid_28",
            "telecoms_newgrid_20",
        ],
        generation_protocol_ids=["controlled_v2_5000"],
        run_kind="batch",
        analysis_tier="exploratory",
    )
    assert len(tags["domains"]) <= 63
    assert "," not in tags["domains"]
    assert re.fullmatch(r"[A-Za-z0-9._-]+", tags["domains"])


def test_analysis_tier_is_explicit_and_closed_set():
    assert validate_analysis_tier("exploratory") == "exploratory"
    assert validate_analysis_tier("definitive") == "definitive"
    assert validate_analysis_tier(" definitive ") == "definitive"
    for value in ("", "pilot", None):
        with pytest.raises(ValueError, match="analysis_tier"):
            validate_analysis_tier(value)


def test_billing_rows_aggregate_without_float_rounding():
    rows = [
        {
            "object_id": "ap-one",
            "description": "spec-gap-qwen3-32b",
            "environment": "main",
            "interval_start": "2026-07-31T10:00:00+00:00",
            "cost": "1.00000001",
            "cost_by_resource": {"H200": "0.99", "CPU": "0.01000001"},
            "tags": {"project": "spec-gap"},
        },
        {
            "object_id": "ap-one",
            "description": "spec-gap-qwen3-32b",
            "environment": "main",
            "interval_start": "2026-07-31T11:00:00+00:00",
            "cost": "2.00000002",
            "cost_by_resource": {"H200": "1.98", "CPU": "0.02000002"},
            "tags": {"project": "spec-gap"},
        },
    ]

    apps, summary = aggregate_billing_rows(rows)

    assert len(apps) == 1
    assert apps[0]["metered_cost_usd"] == "3.00000003"
    assert apps[0]["cost_by_resource_usd"] == {
        "CPU": "0.03000003",
        "H200": "2.97",
    }
    assert summary["metered_cost_usd"] == "3.00000003"


def test_billing_decimal_rejects_negative_or_non_finite_values():
    for value in ("-1", "NaN", "Infinity"):
        with pytest.raises(ValueError):
            as_decimal(value, "cost")

    assert as_decimal("0.10", "cost") == Decimal("0.10")


def test_billing_rows_reject_mid_run_retagging():
    rows = [
        {
            "object_id": "ap-one",
            "description": "spec-gap-qwen3-32b",
            "environment": "main",
            "interval_start": "2026-07-31T10:00:00+00:00",
            "cost": "1.00",
            "cost_by_resource": {"H200": "1.00"},
            "tags": {"project": "spec-gap", "run_kind": "batch"},
        },
        {
            "object_id": "ap-one",
            "description": "spec-gap-qwen3-32b",
            "environment": "main",
            "interval_start": "2026-07-31T11:00:00+00:00",
            "cost": "1.00",
            "cost_by_resource": {"H200": "1.00"},
            "tags": {"project": "spec-gap", "run_kind": "single"},
        },
    ]

    with pytest.raises(ValueError, match="tags changed within App ap-one"):
        aggregate_billing_rows(rows)


def test_protocol_summary_rejects_missing_agent_event_path():
    namespace = runpy.run_path(
        PROTOCOL_SUMMARY_SCRIPT,
        run_name="spec_gap_protocol_summary_test",
    )

    malformed_records = (
        {},
        {"trajectory_trace": {}},
        {"trajectory_trace": {"full_events": {}}},
    )
    for record in malformed_records:
        with pytest.raises(ValueError, match="trajectory_trace"):
            namespace["_agent_turns"](record)


def test_protocol_summary_audits_seed_chain_as_one_shot_sampling():
    namespace = runpy.run_path(
        PROTOCOL_SUMMARY_SCRIPT,
        run_name="spec_gap_protocol_seed_test",
    )
    record = {
        "trajectory_id": "trajectory-one",
        "model": {"seed": 0, "decoding_settings": {"seed": 0}},
        "trajectory_trace": {
            "full_events": [{
                "type": "agent_turn",
                "model_execution_metadata": {
                    "generation_settings": {"seed": 0},
                },
            }],
        },
    }

    audit = namespace["_sampling_reproducibility"]([record])

    assert audit["recorded_seeds"] == [0]
    assert audit["model_turn_count_checked"] == 1
    assert audit["trajectory_decoding_and_saved_turn_seeds_match"] is True
    assert audit["bit_identical_stochastic_rerun_guaranteed"] is False
    assert audit["analysis_unit"] == "saved_one_shot_sample"

    record["trajectory_trace"]["full_events"][0][
        "model_execution_metadata"
    ]["generation_settings"]["seed"] = 1
    with pytest.raises(ValueError, match="saved agent-turn seed"):
        namespace["_sampling_reproducibility"]([record])


def test_protocol_summary_rejects_mixed_saved_analysis_tiers():
    namespace = runpy.run_path(
        PROTOCOL_SUMMARY_SCRIPT,
        run_name="spec_gap_protocol_tier_test",
    )
    record = {
        "trajectory_id": "trajectory-one",
        "trajectory_trace": {
            "full_events": [
                {
                    "type": "agent_turn",
                    "model_execution_metadata": {
                        "analysis_tier": "exploratory"
                    },
                },
                {
                    "type": "agent_turn",
                    "model_execution_metadata": {
                        "analysis_tier": "definitive"
                    },
                },
            ],
        },
    }

    with pytest.raises(ValueError, match="mix analysis tiers"):
        namespace["_saved_analysis_tier"](record)


def test_billing_reconciliation_explains_modal_authentication_failure():
    namespace = runpy.run_path(
        BILLING_RECONCILIATION_SCRIPT,
        run_name="spec_gap_billing_reconciliation_test",
    )

    class FakeAuthError(Exception):
        pass

    class FakeBilling:
        def report(self, **kwargs):
            raise FakeAuthError("missing credentials")

        def summary(self, billing_cycle):
            raise AssertionError("summary should not run after auth failure")

    class FakeWorkspace:
        @staticmethod
        def from_context():
            return SimpleNamespace(billing=FakeBilling())

    fake_modal = SimpleNamespace(
        Workspace=FakeWorkspace,
        exception=SimpleNamespace(AuthError=FakeAuthError),
    )
    now = datetime.now(timezone.utc)

    with pytest.raises(SystemExit, match="Modal authentication failed.*modal setup"):
        namespace["_load_modal_billing"](
            fake_modal,
            start=now,
            end=now,
            billing_cycle="2026-07",
        )


def _billing_row(app_id: str, tier: str | None) -> SimpleNamespace:
    tags = {"project": "spec-gap"}
    if tier is not None:
        tags["analysis_tier"] = tier
    return SimpleNamespace(
        object_id=app_id,
        description="spec-gap-qwen3-32b",
        environment_name="main",
        interval_start=datetime(2026, 7, 31, tzinfo=timezone.utc),
        cost=Decimal("1.00"),
        cost_by_resource={"H200": Decimal("1.00")},
        tags=tags,
    )


def test_billing_reconciliation_selects_one_analysis_tier_only():
    namespace = runpy.run_path(
        BILLING_RECONCILIATION_SCRIPT,
        run_name="spec_gap_billing_tier_test",
    )
    selected = namespace["_normalized_modal_rows"](
        [
            _billing_row("ap-explore", "exploratory"),
            _billing_row("ap-definitive", "definitive"),
            _billing_row("ap-legacy", None),
        ],
        project_tag="spec-gap",
        analysis_tier="definitive",
    )

    assert [row["object_id"] for row in selected] == ["ap-definitive"]
    assert selected[0]["tags"]["analysis_tier"] == "definitive"


def test_billing_reconciliation_rejects_mid_app_tier_changes():
    namespace = runpy.run_path(
        BILLING_RECONCILIATION_SCRIPT,
        run_name="spec_gap_billing_retag_test",
    )
    with pytest.raises(ValueError, match="analysis_tier changed"):
        namespace["_normalized_modal_rows"](
            [
                _billing_row("ap-retagged", "exploratory"),
                _billing_row("ap-retagged", "definitive"),
            ],
            project_tag="spec-gap",
            analysis_tier="definitive",
        )


def test_local_cost_estimate_filters_checkpoint_analysis_tier(tmp_path):
    namespace = runpy.run_path(
        BILLING_RECONCILIATION_SCRIPT,
        run_name="spec_gap_local_cost_tier_test",
    )
    for index, tier in enumerate(("definitive", "exploratory", None)):
        path = tmp_path / f"run-{index}" / "step_001.json"
        path.parent.mkdir(parents=True)
        runtime = {} if tier is None else {"analysis_tier": tier}
        path.write_text(json.dumps({
            "cost_metadata": {
                "modal_input_id": f"in-{index}",
                "estimated_h200_cost_usd": "1.25",
                "runtime_metadata": runtime,
            },
        }))

    result = namespace["_local_estimates"](
        tmp_path,
        analysis_tier="definitive",
    )

    assert result["unique_modal_input_count"] == 1
    assert result["estimated_model_turn_h200_cost_usd"] == "1.25"
    assert result["excluded_other_tier_checkpoint_count"] == 1
    assert result["excluded_unclassified_checkpoint_count"] == 1
