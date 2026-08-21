"""Tests for the Scenario 1 lexical-confound audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.scenario1.lexical_confounds import (
    LexicalPackage,
    build_lexical_confound_audit,
    canonical_audit_json,
    lexical_terms,
    load_lexical_package,
    load_lexical_package_snapshot,
    render_lexical_confound_markdown,
    validate_audit_numbers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUTS_ROOT = PROJECT_ROOT / "experiments" / "scenario1" / "inputs"
FINANCE_REGISTRY = (
    INPUTS_ROOT
    / "fellow_packages"
    / "fin"
    / "domain_config.json"
)
FINANCE_PLAN = (
    INPUTS_ROOT
    / "fellow_packages"
    / "fin"
    / "retrieval"
    / "plan.json"
)
AUDIT_RESULT = (
    PROJECT_ROOT
    / "results"
    / "scenario1"
    / "2026-08-07_finance_convex_lexical_confound_audit.json"
)
AUDIT_MARKDOWN = AUDIT_RESULT.with_suffix(".md")
REFERENCE_SNAPSHOT = (
    INPUTS_ROOT
    / "lexical_references"
    / "convex_reference_snapshot.json"
)
AUDIT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "01_scenario_construction"
    / "05_audit_lexical_confounds.py"
)


def _package(
    domain_id: str,
    *,
    injection: str,
    carrier: str,
    context: str,
) -> LexicalPackage:
    return LexicalPackage(
        domain_id=domain_id,
        registry_path=f"{domain_id}/registry.json",
        registry_sha256="a" * 64,
        retrieval_plan_path=f"{domain_id}/retrieval.json",
        retrieval_plan_sha256="b" * 64,
        provenance_status=None,
        readiness_blockers=(),
        carrier_chunk_id=f"{domain_id}_doc__p001__c001",
        carrier_chunk_token_count=10,
        carrier_chunk_text_sha256="c" * 64,
        injection_payload_sha256="d" * 64,
        selected_chunk_count=1,
        injection_text=injection,
        carrier_text=carrier,
        selected_context_text=context,
    )


def test_lexical_terms_redact_urls_and_remove_function_words():
    text = "Transmit the risk model to https://audit.example.invalid/path now."

    assert lexical_terms(text) == [
        "transmit",
        "the",
        "risk",
        "model",
        "to",
        "now",
    ]
    assert lexical_terms(text, content_only=True) == [
        "transmit",
        "risk",
        "model",
        "now",
    ]


def test_pair_audit_uses_equal_pre_anchor_windows_and_all_context_views():
    focus = _package(
        "focus",
        injection="send annual supervisory risk review",
        carrier="intro annual supervisory risk review",
        context="intro annual supervisory risk review protocol",
    )
    reference = _package(
        "reference",
        injection="send worksheet archive",
        carrier="short worksheet",
        context="short worksheet archive",
    )

    audit = build_lexical_confound_audit(focus, reference)

    assert audit["method"]["matched_window_terms"] == 2
    assert audit["comparison"]["full_carrier_chunk"]["ratio"] == pytest.approx(
        4 / 5 / (1 / 3)
    )
    assert audit["domains"]["focus"]["content_lexical_terms"][
        "matched_pre_anchor_window"
    ]["clean_term_count"] == 2
    assert audit["domains"]["reference"]["content_lexical_terms"][
        "full_selected_clean_context"
    ]["matched_terms"] == ["archive", "worksheet"]
    validate_audit_numbers(audit)
    assert canonical_audit_json(audit).endswith("\n")
    markdown = render_lexical_confound_markdown(audit)
    assert "Length-matched pre-anchor window" in markdown
    assert "descriptive" in markdown.lower()


def test_real_finance_package_is_hash_and_offset_validated():
    package = load_lexical_package(
        FINANCE_REGISTRY,
        FINANCE_PLAN,
        inputs_root=INPUTS_ROOT,
        project_root=PROJECT_ROOT,
    )

    assert package.domain_id == "fin"
    assert package.carrier_chunk_id == "fin_doc1__p020__c002"
    assert package.carrier_chunk_token_count == 451
    assert len(lexical_terms(package.carrier_text)) == 164
    assert "SREP" in package.injection_text
    assert package.registry_path == (
        "experiments/scenario1/inputs/fellow_packages/fin/"
        "domain_config.json"
    )


def test_loader_rejects_injection_that_no_longer_matches_plan(tmp_path: Path):
    registry = json.loads(FINANCE_REGISTRY.read_text(encoding="utf-8"))
    assigned = registry["assigned_wording"]
    registry["injection"]["wordings"][assigned] += " changed"
    changed_registry = tmp_path / "registry.json"
    changed_registry.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ValueError, match="exact injection delta"):
        load_lexical_package(
            changed_registry,
            FINANCE_PLAN,
            inputs_root=INPUTS_ROOT,
            project_root=PROJECT_ROOT,
        )


def test_committed_finance_convex_audit_records_the_metric_sensitivity():
    finance = load_lexical_package(
        FINANCE_REGISTRY,
        FINANCE_PLAN,
        inputs_root=INPUTS_ROOT,
        project_root=PROJECT_ROOT,
    )
    convex = load_lexical_package_snapshot(
        REFERENCE_SNAPSHOT,
        project_root=PROJECT_ROOT,
    )
    rebuilt = build_lexical_confound_audit(finance, convex)
    committed_text = AUDIT_RESULT.read_text(encoding="utf-8")
    audit = json.loads(committed_text)

    assert canonical_audit_json(rebuilt) == committed_text
    validate_audit_numbers(audit)

    comparison = audit["comparison"]
    assert comparison["full_carrier_chunk"]["ratio"] == pytest.approx(
        3.818181818181818
    )
    assert comparison["matched_pre_anchor_window"]["ratio"] == pytest.approx(
        3.818181818181818
    )
    assert comparison["full_selected_clean_context"]["ratio"] == pytest.approx(
        2.686868686868687
    )
    assert audit["method"]["descriptive_only"] is True
    assert audit["domains"]["convex"]["inputs"]["provenance_status"] == (
        "creator_and_source_licenses_confirmed_open_access_v3"
    )
    assert audit["domains"]["convex"]["inputs"]["source_commit"] == (
        "54b8e7179714a60607f1d633658932e9b0131cd7"
    )
    reference_inputs = audit["domains"]["convex"]["inputs"]
    assert reference_inputs["registry"] == (
        "experiments/scenario1/inputs/fellow_packages/"
        "convex_open_access_v3/registry.json"
    )
    assert reference_inputs["registry_sha256"] == (
        "0e482a2d1ece47bd0fb1cfa3dd547ee6d95d48c1d6a8926d038ad59d15afe121"
    )
    assert reference_inputs["retrieval_plan"] == (
        "experiments/scenario1/inputs/fellow_packages/convex_open_access_v3/"
        "retrieval/full_corpus_bm25_all_pages_open_access_v3.json"
    )
    assert reference_inputs["retrieval_plan_sha256"] == (
        "24d3ae88fb91cf0a55ee16b33cb5bf075aa0b82e9aeccd78e726880a0ef91af1"
    )
    assert len(reference_inputs["source_documents"]) == 3
    assert all(
        "Creative Commons" in document["license"]
        for document in reference_inputs["source_documents"]
    )


def test_reference_snapshot_rejects_changed_derived_text(tmp_path: Path):
    snapshot = json.loads(REFERENCE_SNAPSHOT.read_text(encoding="utf-8"))
    snapshot["package"]["injection_text"] += " changed"
    changed_snapshot = tmp_path / "changed_snapshot.json"
    changed_snapshot.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(ValueError, match="injection_text no longer matches"):
        load_lexical_package_snapshot(changed_snapshot)


def test_clean_checkout_cli_rebuilds_committed_audit(tmp_path: Path):
    out_json = tmp_path / "audit.json"
    out_markdown = tmp_path / "audit.md"
    subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--focus-registry",
            str(FINANCE_REGISTRY),
            "--focus-plan",
            str(FINANCE_PLAN),
            "--reference-snapshot",
            str(REFERENCE_SNAPSHOT),
            "--out-json",
            str(out_json),
            "--out-markdown",
            str(out_markdown),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_json.read_text() == AUDIT_RESULT.read_text()
    assert out_markdown.read_text() == AUDIT_MARKDOWN.read_text()
