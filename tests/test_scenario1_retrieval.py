"""Controlled full-corpus retrieval tests for Scenario 1."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from src.infrastructure.qwen_modal import MODEL_ID, MODEL_REVISION
from src.scenario1 import generator
from src.scenario1.retrieval import (
    build_retrieval_plan,
    canonical_plan_sha256,
    detect_single_insertion,
    load_retrieval_plan,
    materialize_retrieval,
    sha256_text,
    validate_retrieval_plan,
)
from src.scenario1.validator import validate_payload


AIHC_REGISTRY_PATH = (
    Path("experiments")
    / "scenario1"
    / "inputs"
    / "fellow_packages"
    / "aihc"
    / "domain_config.json"
)


def test_file_metadata_does_not_change_retrieval_semantic_hash():
    plan = {"schema_version": "example", "selection": {"ids": ["a"]}}
    annotated = {
        "_file_info": {"renamed_from": "old.json", "purpose": "test"},
        **plan,
    }
    assert canonical_plan_sha256(plan) == canonical_plan_sha256(annotated)


class _Encoding:
    def __init__(self, text: str):
        matches = list(re.finditer(r"\S+", text))
        self.ids = list(range(len(matches)))
        self.offsets = [(match.start(), match.end()) for match in matches]


class _WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return _Encoding(text)


@pytest.fixture(scope="module")
def aihc_registry():
    return generator.load_registry(AIHC_REGISTRY_PATH)


@pytest.fixture(scope="module")
def aihc_plan(aihc_registry):
    path = generator.INPUTS / aihc_registry["retrieval"]["plan_file"]
    return load_retrieval_plan(path)


@pytest.fixture(scope="module")
def aihc_pair(aihc_registry):
    return (
        generator.build_record(aihc_registry, "2-hop", "clean"),
        generator.build_record(aihc_registry, "2-hop", "injected"),
    )


def test_plan_builder_detects_one_exact_insertion_and_never_ranks_it():
    documents = [
        {
            "doc_id": "carrier",
            "title": "Carrier",
            "role": "injection_carrier",
            "text": "metric study design anchor",
        },
        {
            "doc_id": "benign_1",
            "title": "Benign one",
            "role": "benign",
            "text": "metric performance validation",
        },
        {
            "doc_id": "benign_2",
            "title": "Benign two",
            "role": "benign",
            "text": "reporting standard metric",
        },
    ]
    plan = build_retrieval_plan(
        documents=documents,
        injected_carrier_text="metric study design TRUSTED NOTE anchor",
        injection_payload="TRUSTED NOTE",
        query="metric design performance reporting",
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        tokenizer_sha256="a" * 64,
        chunk_tokens=20,
        overlap_tokens=0,
        document_token_budget=20,
        context_window_tokens=40,
        max_new_tokens=5,
        non_document_reserve_tokens=5,
    )

    validate_retrieval_plan(
        plan,
        documents,
        injection_payload="TRUSTED NOTE",
    )
    assert plan["canonical_ranking_treatment"] == "clean"
    assert plan["injection_mapping"]["ranking_used_injection_text"] is False
    assert plan["injection_mapping"]["insertion_delta"] == "TRUSTED NOTE "
    assert plan["selection"]["selected_chunk_count"] == 3
    assert plan["injection_mapping"]["carrier_chunk_retention"] == {
        "policy": "natural_only",
        "selection_origin": "natural_clean_rank",
        "selected_from_clean_text": True,
        "replaced_chunk_ids": [],
    }


def test_controlled_retention_replaces_one_clean_ranked_carrier_chunk():
    documents = [
        {
            "doc_id": "carrier",
            "title": "Carrier",
            "role": "injection_carrier",
            "text": "target target target target\fanchor quiet words",
        },
        {
            "doc_id": "benign_1",
            "title": "Benign one",
            "role": "benign",
            "text": "target benign one",
        },
        {
            "doc_id": "benign_2",
            "title": "Benign two",
            "role": "benign",
            "text": "target benign two",
        },
    ]
    build_kwargs = {
        "documents": documents,
        "injected_carrier_text": (
            "target target target target\fanchor TRUSTED NOTE quiet words"
        ),
        "injection_payload": "TRUSTED NOTE",
        "query": "target",
        "tokenizer": _WhitespaceTokenizer(),
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_sha256": "a" * 64,
        "chunk_tokens": 4,
        "overlap_tokens": 0,
        "document_token_budget": 12,
        "document_token_budgets": {
            "carrier": 4,
            "benign_1": 4,
            "benign_2": 4,
        },
        "context_window_tokens": 30,
        "max_new_tokens": 5,
        "non_document_reserve_tokens": 5,
    }

    with pytest.raises(ValueError, match="require_clean_anchor"):
        build_retrieval_plan(**build_kwargs)

    plan = build_retrieval_plan(
        **build_kwargs,
        carrier_chunk_retention_policy="require_clean_anchor",
    )
    validate_retrieval_plan(
        plan,
        documents,
        injection_payload="TRUSTED NOTE",
    )

    mapping = plan["injection_mapping"]
    retention = mapping["carrier_chunk_retention"]
    assert retention == {
        "policy": "require_clean_anchor",
        "selection_origin": "controlled_clean_anchor",
        "selected_from_clean_text": True,
        "replaced_chunk_ids": ["carrier__p001__c001"],
    }
    assert mapping["selected_carrier_chunk_id"] == "carrier__p002__c001"
    assert plan["selection"]["selected_token_count"] == 9

    clean_documents, _, clean_trace = materialize_retrieval(
        plan=plan,
        documents=documents,
        treatment="clean",
        injection_payload="TRUSTED NOTE",
    )
    injected_documents, _, injected_trace = materialize_retrieval(
        plan=plan,
        documents=documents,
        treatment="injected",
        injection_payload="TRUSTED NOTE",
    )
    assert (
        clean_trace["selected_chunk_ids"]
        == injected_trace["selected_chunk_ids"]
    )
    assert all(
        "TRUSTED NOTE" not in document["text"]
        for document in clean_documents
    )
    assert sum(
        document["text"].count("TRUSTED NOTE")
        for document in injected_documents
    ) == 1


def test_aihc_plan_indexes_every_page_of_every_full_source(
    aihc_registry, aihc_plan
):
    documents = generator.load_documents(aihc_registry)
    payload = aihc_registry["injection"]["wordings"][
        aihc_registry["assigned_wording"]
    ]
    validate_retrieval_plan(
        aihc_plan,
        documents,
        injection_payload=payload,
    )

    assert sum(
        source["page_count"] for source in aihc_plan["source_documents"]
    ) == 38
    assert aihc_plan["profile_id"] == "full_corpus_bm25_balanced_gen5000_v2"
    assert aihc_plan["selection"]["candidate_chunk_count"] == 90
    assert aihc_plan["selection"]["eligible_candidate_chunk_count"] == 65
    assert aihc_plan["selection"]["excluded_candidate_chunk_count"] == 25
    assert aihc_plan["selection"]["selected_chunk_count"] == 37
    assert aihc_plan["selection"]["selected_token_count"] == 27643
    assert aihc_plan["evidence_eligibility"] == {
        "determined_from": "clean_source_section_type",
        "all_pages_remain_indexed": True,
        "non_evidence_pages": {
            "aihc_doc1": [
                {
                    "page_number": 7,
                    "reason": (
                        "Reference list and publication-license material."
                    ),
                },
                {
                    "page_number": 8,
                    "reason": (
                        "Reference list and publication-license material."
                    ),
                },
            ],
            "aihc_doc2": [
                {
                    "page_number": page_number,
                    "reason": (
                        "Reference list, publication metadata, and "
                        "supplementary-access notes."
                    ),
                }
                for page_number in (14, 15, 16)
            ],
            "aihc_doc3": [
                {
                    "page_number": page_number,
                    "reason": (
                        "Contributor acknowledgements and reference list."
                    ),
                }
                for page_number in (12, 13, 14)
            ],
        },
    }
    verification = aihc_plan["source_pdf_verification"]
    assert verification["status"] == "verified"
    assert {
        item["doc_id"] for item in verification["documents"]
    } == {document["doc_id"] for document in documents}
    for item in verification["documents"]:
        assert item["matches_indexed_text_fixture"] is True
        assert item["pdf_sha256"] == aihc_registry["provenance"][
            "source_pdf_sha256"
        ][item["filename"]]
    injected_pdf = verification["injected_carrier"]
    assert injected_pdf["pdf_sha256"] == aihc_registry["provenance"][
        "source_pdf_sha256"
    ][injected_pdf["filename"]]
    for source in aihc_plan["source_documents"]:
        assert source["indexed_page_numbers"] == list(
            range(1, source["page_count"] + 1)
        )
    assert all(
        candidate["evidence_eligible"] is True
        for candidate in aihc_plan["candidate_chunks"]
        if candidate["selected"]
    )


def test_aihc_selection_uses_balanced_per_document_caps(aihc_plan):
    budgets = aihc_plan["budget"]["document_token_budgets"]
    assert budgets == {
        "aihc_doc1": 10000,
        "aihc_doc2": 9000,
        "aihc_doc3": 9000,
    }
    selected = [
        candidate
        for candidate in aihc_plan["candidate_chunks"]
        if candidate["selected"]
    ]
    chunk_counts = {
        doc_id: sum(candidate["doc_id"] == doc_id for candidate in selected)
        for doc_id in budgets
    }
    token_counts = {
        doc_id: sum(
            candidate["token_count"]
            for candidate in selected
            if candidate["doc_id"] == doc_id
        )
        for doc_id in budgets
    }
    assert chunk_counts == {
        "aihc_doc1": 12,
        "aihc_doc2": 14,
        "aihc_doc3": 11,
    }
    assert token_counts == {
        "aihc_doc1": 9936,
        "aihc_doc2": 8847,
        "aihc_doc3": 8860,
    }
    assert all(
        token_counts[doc_id] <= budget
        for doc_id, budget in budgets.items()
    )
    assert max(token_counts.values()) - min(token_counts.values()) <= 1100


def test_aihc_selection_is_clean_ranked_and_retains_the_carrier_location(
    aihc_plan,
):
    mapping = aihc_plan["injection_mapping"]
    selected_ids = aihc_plan["selection"]["render_order_chunk_ids"]
    candidate = next(
        item for item in aihc_plan["candidate_chunks"]
        if item["chunk_id"] == mapping["selected_carrier_chunk_id"]
    )

    assert aihc_plan["canonical_ranking_treatment"] == "clean"
    assert mapping["ranking_used_injection_text"] is False
    assert mapping["selected_carrier_chunk_id"] in selected_ids
    assert candidate["bm25_rank"] == 31
    assert candidate["selected"] is True
    assert mapping["insertion_delta_sha256"] == sha256_text(
        mapping["insertion_delta"]
    )


def test_aihc_clean_and_injected_views_use_identical_chunks(aihc_pair):
    clean, injected = aihc_pair
    assert (
        clean["retrieval_trace"]["selected_chunk_ids"]
        == injected["retrieval_trace"]["selected_chunk_ids"]
    )
    assert (
        clean["retrieval_trace"]["plan_sha256"]
        == injected["retrieval_trace"]["plan_sha256"]
    )

    differing = []
    for clean_document, injected_document in zip(
        clean["document_set"]["documents"],
        injected["document_set"]["documents"],
    ):
        assert clean_document["doc_id"] == injected_document["doc_id"]
        assert clean_document["retrieval"] == injected_document["retrieval"]
        if clean_document["text"] != injected_document["text"]:
            differing.append(clean_document["doc_id"])
            offset, delta = detect_single_insertion(
                clean_document["text"],
                injected_document["text"],
            )
            assert offset >= 0
            assert delta.count(injected["injection"]["injected_text"]) == 1
    assert differing == [injected["injection"]["injection_source_id"]]


def test_aihc_payload_occurs_once_only_in_injected_worker_view(
    aihc_registry, aihc_pair
):
    clean, injected = aihc_pair
    payload = aihc_registry["injection"]["wordings"][
        aihc_registry["assigned_wording"]
    ]
    assert sum(
        document["text"].count(payload)
        for document in clean["document_set"]["documents"]
    ) == 0
    assert sum(
        document["text"].count(payload)
        for document in injected["document_set"]["documents"]
    ) == 1

    worker = next(
        event for event in injected["trajectory_trace"]["full_events"]
        if event.get("agent_id") == "worker_1"
    )
    request = generator.build_generation_request(
        injected,
        worker,
        thinking_mode="off",
    )
    assert sum(
        message["content"].count(payload) for message in request["messages"]
    ) == 1


def test_aihc_exact_qwen_context_preflight_has_headroom(
    aihc_registry, aihc_plan, aihc_pair
):
    path = (
        generator.INPUTS
        / aihc_registry["retrieval"]["context_preflight_file"]
    )
    preflight = json.loads(path.read_text())
    assert preflight["retrieval_plan_sha256"] == (
        aihc_pair[0]["retrieval_trace"]["plan_sha256"]
    )
    assert preflight["tokenizer_json_sha256"] == (
        aihc_plan["tokenizer"]["tokenizer_json_sha256"]
    )
    assert {
        (case["treatment"], case["thinking_mode"])
        for case in preflight["cases"]
    } == {
        ("clean", "off"),
        ("clean", "on"),
        ("injected", "off"),
        ("injected", "on"),
    }
    assert min(case["headroom_tokens"] for case in preflight["cases"]) == 1914
    assert aihc_pair[0]["retrieval_trace"][
        "minimum_preflight_headroom_tokens"
    ] == 1914


def test_aihc_retrieval_records_pass_schema_and_semantic_checks(aihc_registry):
    for depth in ("2-hop", "3-hop"):
        for treatment in ("clean", "injected"):
            record = generator.build_record(aihc_registry, depth, treatment)
            assert validate_payload(record) == []


def test_plan_validation_rejects_a_changed_source_slice(
    aihc_registry, aihc_plan
):
    documents = generator.load_documents(aihc_registry)
    payload = aihc_registry["injection"]["wordings"][
        aihc_registry["assigned_wording"]
    ]
    tampered = copy.deepcopy(aihc_plan)
    tampered["candidate_chunks"][0]["text_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="no longer matches"):
        validate_retrieval_plan(
            tampered,
            documents,
            injection_payload=payload,
        )


def test_semantic_validator_rejects_retrieval_selection_drift(aihc_pair):
    record = copy.deepcopy(aihc_pair[0])
    record["retrieval_trace"]["selected_chunk_ids"][0] = "changed"
    assert any(
        "INV-11" in error for error in validate_payload(record)
    )


def test_semantic_validator_rejects_selected_non_evidence_page(aihc_pair):
    record = copy.deepcopy(aihc_pair[0])
    record["retrieval_trace"]["selected_chunks"][0]["doc_id"] = "aihc_doc1"
    record["retrieval_trace"]["selected_chunks"][0]["page_number"] = 7
    assert any(
        "INV-11" in error for error in validate_payload(record)
    )
