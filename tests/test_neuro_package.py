"""Review invariants for the active publication-backed Neuro package.

Former filename: ``tests/test_neuro_v2_package.py``.
Original date: not encoded in the former filename.
Renamed on: 2026-08-10.
Purpose: test the active package while protocol provenance stays in the data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ROOT
    / "experiments"
    / "scenario1"
    / "inputs"
    / "fellow_packages"
    / "neuro"
)
REGISTRY_PATH = PACKAGE_ROOT / "domain_config.json"
PLAN_PATH = PACKAGE_ROOT / "retrieval" / "plan.json"
SUMMARY_PATH = (
    ROOT
    / "results"
    / "scenario1"
    / "2026-07-31_neuro_full_matrix_gen5000_v2_readable_summary.md"
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
MATH_LATEX_PATTERN = re.compile(r"[\\^_{}=<>±×÷−+*|∑∫√≤≥≈%]")
CITATION_PATTERN = re.compile(
    r"\[\s*\d+(?:\s*[-,–]\s*\d+)*\s*\]"
    r"|\b(?:19|20)\d{2}[a-z]?\b"
    r"|\bet al\."
    r"|^\s*\d+\s+[A-Z]"
    r"|https?://doi\.org/"
    r"|\bdoi\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _texture_counts(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "approx_tokens": len(TOKEN_PATTERN.findall(text)),
        "math_latex_like_characters": len(MATH_LATEX_PATTERN.findall(text)),
        "citation_marker_occurrences": len(CITATION_PATTERN.findall(text)),
    }


def test_neuro_provenance_and_carrier_are_unambiguous():
    registry = _load_json(REGISTRY_PATH)
    plan = _load_json(PLAN_PATH)
    provenance = registry["provenance"]
    runtime_ids = [slot["doc_id"] for slot in registry["document_slots"]]

    assert runtime_ids == ["neuro_doc1", "neuro_doc2", "neuro_doc3"]
    assert [
        item["doc_id"] for item in provenance["source_documents"]
    ] == runtime_ids
    assert all(item["doi"] and item["license"] for item in provenance["source_documents"])

    resolution = provenance["registry_resolution"]
    carrier = next(
        slot for slot in registry["document_slots"]
        if slot["role"] == "injection_carrier"
    )
    assert carrier["doc_id"] == resolution["authoritative_carrier_doc_id"]
    assert carrier["doc_id"] == plan["injection_mapping"]["carrier_doc_id"]
    assert plan["injection_mapping"]["selected_carrier_chunk_id"].startswith(
        f"{carrier['doc_id']}__"
    )
    assert resolution["legacy_synthetic_registry"]["status"].startswith(
        "obsolete_unmerged_draft"
    )


def test_neuro_nc_nd_document_has_an_explicit_license_boundary():
    registry = _load_json(REGISTRY_PATH)
    neuro_doc3 = next(
        item for item in registry["provenance"]["source_documents"]
        if item["doc_id"] == "neuro_doc3"
    )
    notice = (PACKAGE_ROOT / "LICENSE_NOTICE.md").read_text()

    assert "NonCommercial-NoDerivs" in neuro_doc3["license"]
    assert "not relicensed" in neuro_doc3["repository_license_boundary"]
    assert "MIT License" in notice
    assert "non-commercial" in notice
    assert "must not distribute modified or adapted content" in notice


def test_neuro_summary_enumerates_the_complete_eight_condition_matrix():
    summary = SUMMARY_PATH.read_text()
    reported_pairs = set(
        re.findall(
            r"^\| (Off|On) \| (2-hop|3-hop) \| `clean` \| `resisted` \|$",
            summary,
            re.MULTILINE,
        )
    )

    assert reported_pairs == {
        ("Off", "2-hop"),
        ("Off", "3-hop"),
        ("On", "2-hop"),
        ("On", "3-hop"),
    }
    assert len(reported_pairs) * 2 == 8


def test_neuro_v2_carrier_neighborhood_descriptor_is_reproducible():
    registry = _load_json(REGISTRY_PATH)
    plan = _load_json(PLAN_PATH)
    descriptor_path = (
        ROOT
        / "experiments"
        / "scenario1"
        / "inputs"
        / registry["provenance"]["carrier_neighborhood_audit"]
    )
    descriptor = _load_json(descriptor_path)
    source_text = (PACKAGE_ROOT / "documents" / "neuro_doc1_clean.txt").read_text()
    offset = plan["injection_mapping"]["source_char_offset"]
    payload = plan["injection_mapping"]["insertion_delta"]
    tokens = list(TOKEN_PATTERN.finditer(source_text))
    left = [token for token in tokens if token.end() <= offset][-256:]
    right = [token for token in tokens if token.start() >= offset][:256]
    start = left[0].start()
    end = right[-1].end()
    background = source_text[start:end]
    injected = background[: offset - start] + payload + background[offset - start :]

    assert descriptor["source_char_offset"] == offset
    assert descriptor["window"]["clean_source_char_start"] == start
    assert descriptor["window"]["clean_source_char_end"] == end
    assert descriptor["clean_background"] == _texture_counts(background)
    assert descriptor["injected_background_after_payload_removal"] == (
        _texture_counts(background)
    )
    assert descriptor["payload_contribution"] == _texture_counts(payload)
    assert descriptor["injected_total"] == _texture_counts(injected)
