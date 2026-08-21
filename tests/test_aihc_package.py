"""Publication metadata and privacy checks for the active AIHC package.

Former filename: ``tests/test_aihc_v2_package.py``.
Original date: not encoded in the former filename.
Renamed on: 2026-08-10.
Purpose: test the active package without encoding a historical protocol suffix.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.scenario1 import generator


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ROOT
    / "experiments"
    / "scenario1"
    / "inputs"
    / "fellow_packages"
    / "aihc"
)
REGISTRY_PATH = PACKAGE_ROOT / "domain_config.json"

PATIENT_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:patient|subject|participant)[ _-]?(?:id|identifier|number)\b"
    r"|\b(?:mrn|medical record number|date of birth|social security number|ssn)\b"
    r"|\b\d{3}-\d{2}-\d{4}\b",
    re.IGNORECASE,
)
ACCESSION_LIKE_PATTERN = re.compile(
    r"\b(?:NCT\d{8}|ISRCTN\d{8}|EudraCT[- ]?[0-9A-Za-z-]+"
    r"|[A-Za-z]{2,6}[-_]?\d{6,})\b"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_aihc_source_licenses_and_config_scope_are_explicit():
    registry = _load_json(REGISTRY_PATH)
    provenance = registry["provenance"]
    runtime_ids = [slot["doc_id"] for slot in registry["document_slots"]]
    source_documents = provenance["source_documents"]

    assert runtime_ids == ["aihc_doc1", "aihc_doc2", "aihc_doc3"]
    assert [item["doc_id"] for item in source_documents] == runtime_ids
    assert all(item["doi"] for item in source_documents)
    assert all(item["source_url"].startswith("https://doi.org/") for item in source_documents)
    assert {
        item["license"] for item in source_documents
    } == {
        "Creative Commons Attribution 4.0 International (CC BY 4.0)"
    }

    resolution = provenance["registry_resolution"]
    assert resolution["runtime_document_ids"] == runtime_ids
    assert resolution["authoritative_runtime_registry"].endswith(
        "domain_config.json"
    )
    assert resolution["legacy_synthetic_registry"]["status"].startswith(
        "obsolete_unmerged_draft"
    )


def test_aihc_phi_hygiene_covers_both_materialized_arms():
    registry = generator.load_registry(REGISTRY_PATH)
    audit_path = (
        generator.INPUTS / registry["provenance"]["privacy_hygiene"]["audit_file"]
    )
    audit = _load_json(audit_path)
    expected_public_identifiers = {
        item["value"] for item in audit["public_non_patient_identifiers"]
    }

    identifiers_by_treatment = {}
    for treatment in ("clean", "injected"):
        record = generator.build_record(registry, "2-hop", treatment)
        materialized_text = "\n".join(
            document["text"]
            for document in record["document_set"]["documents"]
        )
        assert PATIENT_IDENTIFIER_PATTERN.search(materialized_text) is None
        identifiers_by_treatment[treatment] = set(
            ACCESSION_LIKE_PATTERN.findall(materialized_text)
        )

    assert identifiers_by_treatment == {
        "clean": expected_public_identifiers,
        "injected": expected_public_identifiers,
    }
    assert audit["result"] == {
        "patient_identifier_matches": 0,
        "accession_like_matches_requiring_classification": 1,
        "status": "passed",
    }
