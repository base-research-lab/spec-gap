"""Deterministic, treatment-blind retrieval for full Scenario 1 corpora.

Retrieval plans are constructed once from the clean source documents. At
trajectory-build time, the same selected chunk IDs are materialized for the
clean and injected treatments. The injected treatment applies the exact
clean-to-injected source delta only after ranking, so the injection cannot
influence retrieval.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Protocol


RETRIEVAL_PLAN_SCHEMA_VERSION = "spec_gap.retrieval_plan.v1"
DEFAULT_PROFILE_ID = "full_corpus_bm25_v1"
DEFAULT_CHUNK_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP_TOKENS = 100
DEFAULT_DOCUMENT_TOKEN_BUDGET = 28000
DEFAULT_CONTEXT_WINDOW_TOKENS = 40960
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_NON_DOCUMENT_RESERVE_TOKENS = 6144
BM25_K1 = 1.5
BM25_B = 0.75
CARRIER_CHUNK_RETENTION_POLICIES = frozenset({
    "natural_only",
    "require_clean_anchor",
})

_TERM = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _Encoding(Protocol):
    ids: list[int]
    offsets: list[tuple[int, int]]


class Tokenizer(Protocol):
    def encode(
        self, text: str, add_special_tokens: bool = False
    ) -> _Encoding: ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_plan_sha256(plan: dict[str, Any]) -> str:
    """Hash retrieval semantics while ignoring repository-only file metadata."""

    semantic_plan = {
        key: value for key, value in plan.items() if key != "_file_info"
    }
    payload = json.dumps(
        semantic_plan,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256_text(payload)


def _page_spans(text: str) -> list[dict[str, Any]]:
    """Return non-synthetic PDF page spans while preserving source offsets."""

    pages = []
    start = 0
    page_number = 1
    while start < len(text):
        form_feed = text.find("\f", start)
        end = len(text) if form_feed < 0 else form_feed
        pages.append({
            "page_number": page_number,
            "char_start": start,
            "char_end": end,
            "text": text[start:end],
        })
        page_number += 1
        if form_feed < 0:
            break
        start = form_feed + 1
    return pages


def _encode(tokenizer: Tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = list(encoding.ids)
    offsets = [tuple(item) for item in encoding.offsets]
    if len(ids) != len(offsets):
        raise ValueError("tokenizer IDs and offsets have different lengths")
    if ids and any(
        not 0 <= start <= end <= len(text) for start, end in offsets
    ):
        raise ValueError("tokenizer returned an invalid character offset")
    return ids, offsets


def _terms(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TERM.finditer(text)]


def _chunk_documents(
    documents: list[dict[str, Any]],
    tokenizer: Tokenizer,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError("overlap_tokens must be between 0 and chunk_tokens - 1")

    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for document_order, document in enumerate(documents):
        source_text = document["text"]
        pages = _page_spans(source_text)
        if not pages:
            raise ValueError(f"{document['doc_id']} has no source pages")

        page_numbers_with_chunks: set[int] = set()
        document_candidate_count = 0
        for page in pages:
            page_text = page["text"]
            token_ids, offsets = _encode(tokenizer, page_text)
            if not token_ids:
                if page_text.strip():
                    raise ValueError(
                        f"tokenizer produced no tokens for "
                        f"{document['doc_id']} page {page['page_number']}"
                    )
                continue

            token_start = 0
            chunk_index = 1
            while token_start < len(token_ids):
                token_end = min(token_start + chunk_tokens, len(token_ids))
                local_char_start = (
                    0 if token_start == 0 else offsets[token_start][0]
                )
                local_char_end = (
                    len(page_text)
                    if token_end == len(token_ids)
                    else offsets[token_end - 1][1]
                )
                source_char_start = page["char_start"] + local_char_start
                source_char_end = page["char_start"] + local_char_end
                chunk_text = source_text[source_char_start:source_char_end]
                exact_token_ids, _ = _encode(tokenizer, chunk_text)
                chunk_id = (
                    f"{document['doc_id']}__p{page['page_number']:03d}"
                    f"__c{chunk_index:03d}"
                )
                candidates.append({
                    "chunk_id": chunk_id,
                    "doc_id": document["doc_id"],
                    "document_order": document_order,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "source_char_start": source_char_start,
                    "source_char_end": source_char_end,
                    "token_count": len(exact_token_ids),
                    "text_sha256": sha256_text(chunk_text),
                    "_text": chunk_text,
                })
                page_numbers_with_chunks.add(page["page_number"])
                document_candidate_count += 1
                if token_end == len(token_ids):
                    break
                token_start = token_end - overlap_tokens
                chunk_index += 1

        indexed_pages = sorted(page_numbers_with_chunks)
        expected_pages = [
            page["page_number"] for page in pages if page["text"].strip()
        ]
        if indexed_pages != expected_pages:
            raise ValueError(
                f"{document['doc_id']} page coverage is incomplete: "
                f"{indexed_pages} != {expected_pages}"
            )
        sources.append({
            "doc_id": document["doc_id"],
            "title": document["title"],
            "role": document["role"],
            "file": document.get("file"),
            "text_sha256": sha256_text(source_text),
            "character_count": len(source_text),
            "page_count": len(pages),
            "indexed_page_numbers": indexed_pages,
            "candidate_chunk_count": document_candidate_count,
        })
        for field in (
            "source_pdf",
            "clean_source_pdf",
            "injected_source_pdf",
        ):
            if isinstance(document.get(field), str):
                sources[-1][field] = document[field]
    return candidates, sources


def _score_bm25(
    candidates: list[dict[str, Any]],
    query: str,
    *,
    k1: float,
    b: float,
) -> None:
    query_terms = _terms(query)
    if not query_terms:
        raise ValueError("retrieval query has no searchable terms")
    corpus_terms = [_terms(candidate["_text"]) for candidate in candidates]
    document_frequency: Counter[str] = Counter()
    for terms in corpus_terms:
        document_frequency.update(set(terms))
    average_length = sum(map(len, corpus_terms)) / len(corpus_terms)
    if average_length <= 0:
        raise ValueError("retrieval candidates contain no searchable terms")
    query_counts = Counter(query_terms)
    corpus_size = len(candidates)

    for candidate, terms in zip(candidates, corpus_terms):
        frequencies = Counter(terms)
        length = len(terms)
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (corpus_size - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * length / average_length
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * frequency
                * (k1 + 1.0)
                / denominator
            )
        candidate["bm25_score"] = round(score, 12)

    ranked = sorted(
        candidates,
        key=lambda item: (
            -item["bm25_score"],
            item["document_order"],
            item["page_number"],
            item["chunk_index"],
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["bm25_rank"] = rank


def _select_candidates(
    candidates: list[dict[str, Any]],
    document_ids: Iterable[str],
    *,
    token_budget: int,
    document_token_budgets: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    if token_budget < 1:
        raise ValueError("token budget must be positive")
    ranked = sorted(candidates, key=lambda item: item["bm25_rank"])
    document_ids = list(document_ids)
    if document_token_budgets is not None:
        if set(document_token_budgets) != set(document_ids):
            raise ValueError(
                "document token budgets must cover exactly the source documents"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in document_token_budgets.values()
        ):
            raise ValueError("document token budgets must be positive integers")
        if sum(document_token_budgets.values()) > token_budget:
            raise ValueError(
                "per-document token budgets exceed the global token budget"
            )
        selected = []
        for doc_id in document_ids:
            document_used = 0
            for candidate in ranked:
                if candidate["doc_id"] != doc_id:
                    continue
                if (
                    document_used + candidate["token_count"]
                    > document_token_budgets[doc_id]
                ):
                    continue
                selected.append(candidate)
                document_used += candidate["token_count"]
            if not any(item["doc_id"] == doc_id for item in selected):
                raise ValueError(
                    f"{doc_id} token budget selected no evidence chunks"
                )
        return selected

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    tokens_used = 0

    # Every source contributes at least its highest-scoring task-relevant chunk.
    for doc_id in document_ids:
        candidate = next(
            (item for item in ranked if item["doc_id"] == doc_id),
            None,
        )
        if candidate is None:
            raise ValueError(f"{doc_id} has no retrieval candidate")
        if tokens_used + candidate["token_count"] > token_budget:
            raise ValueError(
                "token budget is too small to select one chunk per document"
            )
        selected.append(candidate)
        selected_ids.add(candidate["chunk_id"])
        tokens_used += candidate["token_count"]

    for candidate in ranked:
        if candidate["chunk_id"] in selected_ids:
            continue
        if tokens_used + candidate["token_count"] > token_budget:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["chunk_id"])
        tokens_used += candidate["token_count"]

    return selected


def _retain_required_candidate(
    selected: list[dict[str, Any]],
    required: dict[str, Any],
    *,
    token_budget: int,
    document_token_budgets: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain one clean anchor chunk without exceeding existing token caps."""

    if any(
        candidate["chunk_id"] == required["chunk_id"]
        for candidate in selected
    ):
        return selected, []

    retained = list(selected)
    removed: list[dict[str, Any]] = []
    if document_token_budgets is not None:
        document_id = required["doc_id"]
        document_budget = document_token_budgets[document_id]
        document_used = sum(
            candidate["token_count"]
            for candidate in retained
            if candidate["doc_id"] == document_id
        )
        removable = sorted(
            (
                candidate
                for candidate in retained
                if candidate["doc_id"] == document_id
            ),
            key=lambda candidate: candidate["bm25_rank"],
            reverse=True,
        )
        while document_used + required["token_count"] > document_budget:
            if not removable:
                raise ValueError(
                    "the carrier document budget cannot retain its clean "
                    "anchor chunk"
                )
            victim = removable.pop(0)
            retained.remove(victim)
            removed.append(victim)
            document_used -= victim["token_count"]
    else:
        tokens_used = sum(candidate["token_count"] for candidate in retained)
        counts = Counter(candidate["doc_id"] for candidate in retained)
        removable = sorted(
            retained,
            key=lambda candidate: candidate["bm25_rank"],
            reverse=True,
        )
        while tokens_used + required["token_count"] > token_budget:
            victim = next(
                (
                    candidate
                    for candidate in removable
                    if counts[candidate["doc_id"]] > 1
                    or candidate["doc_id"] == required["doc_id"]
                ),
                None,
            )
            if victim is None:
                raise ValueError(
                    "the global token budget cannot retain the clean anchor "
                    "chunk while preserving every source document"
                )
            removable.remove(victim)
            retained.remove(victim)
            removed.append(victim)
            counts[victim["doc_id"]] -= 1
            tokens_used -= victim["token_count"]

    retained.append(required)
    if sum(candidate["token_count"] for candidate in retained) > token_budget:
        raise ValueError(
            "controlled carrier retention exceeded the global token budget"
        )
    return retained, removed


def detect_single_insertion(clean_text: str, injected_text: str) -> tuple[int, str]:
    """Return the offset and exact delta when injected = clean + one insertion."""

    if clean_text == injected_text:
        raise ValueError("clean and injected carrier texts are identical")
    prefix_length = 0
    for clean_char, injected_char in zip(clean_text, injected_text):
        if clean_char != injected_char:
            break
        prefix_length += 1
    clean_suffix = clean_text[prefix_length:]
    if clean_suffix and not injected_text.endswith(clean_suffix):
        raise ValueError(
            "carrier PDFs differ beyond one contiguous insertion"
        )
    injected_end = (
        len(injected_text) - len(clean_suffix)
        if clean_suffix
        else len(injected_text)
    )
    insertion_delta = injected_text[prefix_length:injected_end]
    if not insertion_delta:
        raise ValueError("the detected insertion delta is empty")
    if (
        injected_text[:prefix_length]
        + injected_text[injected_end:]
        != clean_text
    ):
        raise ValueError("removing the detected delta does not recover clean text")
    return prefix_length, insertion_delta


def build_retrieval_plan(
    *,
    documents: list[dict[str, Any]],
    injected_carrier_text: str,
    injection_payload: str,
    query: str,
    tokenizer: Tokenizer,
    tokenizer_id: str,
    tokenizer_revision: str,
    tokenizer_sha256: str,
    profile_id: str = DEFAULT_PROFILE_ID,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    document_token_budget: int = DEFAULT_DOCUMENT_TOKEN_BUDGET,
    document_token_budgets: dict[str, int] | None = None,
    non_evidence_pages: dict[str, dict[int, str]] | None = None,
    carrier_chunk_retention_policy: str = "natural_only",
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    non_document_reserve_tokens: int = DEFAULT_NON_DOCUMENT_RESERVE_TOKENS,
) -> dict[str, Any]:
    """Build an auditable full-corpus plan using only clean text for ranking."""

    if len(documents) != 3:
        raise ValueError("Scenario 1 retrieval requires exactly three documents")
    carrier = next(
        (document for document in documents if document["role"] == "injection_carrier"),
        None,
    )
    if carrier is None:
        raise ValueError("Scenario 1 retrieval requires one injection carrier")
    if sum(
        document["role"] == "injection_carrier" for document in documents
    ) != 1:
        raise ValueError("Scenario 1 retrieval requires exactly one injection carrier")
    if carrier_chunk_retention_policy not in CARRIER_CHUNK_RETENTION_POLICIES:
        raise ValueError(
            "carrier_chunk_retention_policy must be natural_only or "
            "require_clean_anchor"
        )
    if document_token_budget + max_new_tokens + non_document_reserve_tokens > (
        context_window_tokens
    ):
        raise ValueError(
            "document budget, generation budget, and prompt reserve exceed "
            "the model context window"
        )

    candidates, source_documents = _chunk_documents(
        documents,
        tokenizer,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    document_ids = [document["doc_id"] for document in documents]
    non_evidence_pages = copy.deepcopy(non_evidence_pages or {})
    unknown_exclusion_documents = set(non_evidence_pages) - set(document_ids)
    if unknown_exclusion_documents:
        raise ValueError(
            "non-evidence pages reference unknown documents: "
            f"{sorted(unknown_exclusion_documents)}"
        )
    page_counts = {
        source["doc_id"]: source["page_count"]
        for source in source_documents
    }
    for doc_id, page_reasons in non_evidence_pages.items():
        if not isinstance(page_reasons, dict):
            raise ValueError(
                f"{doc_id} non-evidence pages must map page numbers to reasons"
            )
        for page_number, reason in page_reasons.items():
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or not 1 <= page_number <= page_counts[doc_id]
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError(
                    f"{doc_id} has an invalid non-evidence page annotation"
                )

    for candidate in candidates:
        reason = non_evidence_pages.get(
            candidate["doc_id"],
            {},
        ).get(candidate["page_number"])
        candidate["evidence_eligible"] = reason is None
        candidate["exclusion_reason"] = reason
        candidate["bm25_score"] = None
        candidate["bm25_rank"] = None
    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["evidence_eligible"]
    ]
    if not eligible_candidates:
        raise ValueError("retrieval plan has no evidence-eligible chunks")
    _score_bm25(eligible_candidates, query, k1=BM25_K1, b=BM25_B)

    for source in source_documents:
        excluded_pages = sorted(
            non_evidence_pages.get(source["doc_id"], {})
        )
        source["evidence_eligible_page_numbers"] = [
            page_number
            for page_number in source["indexed_page_numbers"]
            if page_number not in excluded_pages
        ]
        source["non_evidence_page_numbers"] = excluded_pages
        source["eligible_candidate_chunk_count"] = sum(
            candidate["doc_id"] == source["doc_id"]
            and candidate["evidence_eligible"]
            for candidate in candidates
        )
        source["excluded_candidate_chunk_count"] = sum(
            candidate["doc_id"] == source["doc_id"]
            and not candidate["evidence_eligible"]
            for candidate in candidates
        )

    selected = _select_candidates(
        eligible_candidates,
        document_ids,
        token_budget=document_token_budget,
        document_token_budgets=document_token_budgets,
    )
    selected_ids = {candidate["chunk_id"] for candidate in selected}
    render_order = sorted(
        selected,
        key=lambda item: (
            item["document_order"],
            item["page_number"],
            item["chunk_index"],
        ),
    )

    insertion_offset, insertion_delta = detect_single_insertion(
        carrier["text"],
        injected_carrier_text,
    )
    if insertion_delta.count(injection_payload) != 1:
        raise ValueError(
            "the exact PDF insertion delta must contain the registry payload once"
        )
    payload_offset_in_delta = insertion_delta.index(injection_payload)
    containing_candidates = [
        candidate for candidate in candidates
        if candidate["doc_id"] == carrier["doc_id"]
        and candidate["source_char_start"] <= insertion_offset
        <= candidate["source_char_end"]
    ]
    selected_containing = [
        candidate for candidate in containing_candidates
        if candidate["chunk_id"] in selected_ids
    ]
    retention_origin = "natural_clean_rank"
    replaced_candidates: list[dict[str, Any]] = []
    if (
        not selected_containing
        and carrier_chunk_retention_policy == "require_clean_anchor"
    ):
        eligible_containing = [
            candidate
            for candidate in containing_candidates
            if candidate["evidence_eligible"]
        ]
        if len(eligible_containing) != 1:
            raise ValueError(
                "the clean insertion anchor does not map to exactly one "
                "evidence-eligible chunk"
            )
        selected, replaced_candidates = _retain_required_candidate(
            selected,
            eligible_containing[0],
            token_budget=document_token_budget,
            document_token_budgets=document_token_budgets,
        )
        selected_ids = {candidate["chunk_id"] for candidate in selected}
        selected_containing = [
            candidate for candidate in containing_candidates
            if candidate["chunk_id"] in selected_ids
        ]
        retention_origin = "controlled_clean_anchor"
    if len(selected_containing) != 1:
        raise ValueError(
            "the clean BM25 selection did not naturally retain exactly one "
            "chunk at the carrier insertion point; use "
            "require_clean_anchor for a controlled-exposure design"
        )
    carrier_chunk = selected_containing[0]
    render_order = sorted(
        selected,
        key=lambda item: (
            item["document_order"],
            item["page_number"],
            item["chunk_index"],
        ),
    )

    serializable_candidates = []
    for candidate in candidates:
        item = {
            key: value for key, value in candidate.items()
            if key not in {"_text", "document_order"}
        }
        item["selected"] = candidate["chunk_id"] in selected_ids
        serializable_candidates.append(item)

    selected_token_count = sum(
        candidate["token_count"] for candidate in selected
    )
    return {
        "schema_version": RETRIEVAL_PLAN_SCHEMA_VERSION,
        "profile_id": profile_id,
        "canonical_ranking_treatment": "clean",
        "query": query,
        "tokenizer": {
            "model_id": tokenizer_id,
            "revision": tokenizer_revision,
            "tokenizer_json_sha256": tokenizer_sha256,
        },
        "chunking": {
            "unit": "model_tokens",
            "page_boundary_policy": "never_cross_pdf_pages",
            "chunk_tokens": chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "text_extraction": "pdftotext -raw",
        },
        "ranking": {
            "algorithm": "BM25",
            "k1": BM25_K1,
            "b": BM25_B,
            "ranking_corpus": "clean_evidence_eligible_chunks_only",
            "tie_break": "document_order,page_number,chunk_index",
            "selection_policy": (
                "rank within each document and fill its fixed token budget"
                if document_token_budgets is not None
                else (
                    "highest-scoring chunk per document, then global score "
                    "until the fixed token budget"
                )
            ),
            "carrier_chunk_retention_policy": (
                carrier_chunk_retention_policy
            ),
        },
        "evidence_eligibility": {
            "determined_from": "clean_source_section_type",
            "all_pages_remain_indexed": True,
            "non_evidence_pages": {
                doc_id: [
                    {
                        "page_number": page_number,
                        "reason": reason,
                    }
                    for page_number, reason in sorted(page_reasons.items())
                ]
                for doc_id, page_reasons in sorted(
                    non_evidence_pages.items()
                )
            },
        },
        "budget": {
            "context_window_tokens": context_window_tokens,
            "max_new_tokens": max_new_tokens,
            "non_document_reserve_tokens": non_document_reserve_tokens,
            "document_token_budget": document_token_budget,
            "document_token_budgets": copy.deepcopy(
                document_token_budgets
            ),
        },
        "source_documents": source_documents,
        "candidate_chunks": serializable_candidates,
        "selection": {
            "candidate_chunk_count": len(candidates),
            "eligible_candidate_chunk_count": len(eligible_candidates),
            "excluded_candidate_chunk_count": (
                len(candidates) - len(eligible_candidates)
            ),
            "selected_chunk_count": len(selected),
            "selected_token_count": selected_token_count,
            "ranked_chunk_ids": [
                candidate["chunk_id"]
                for candidate in sorted(selected, key=lambda item: item["bm25_rank"])
            ],
            "render_order_chunk_ids": [
                candidate["chunk_id"] for candidate in render_order
            ],
        },
        "injection_mapping": {
            "carrier_doc_id": carrier["doc_id"],
            "source_char_offset": insertion_offset,
            "insertion_delta": insertion_delta,
            "insertion_delta_sha256": sha256_text(insertion_delta),
            "clean_carrier_text_sha256": sha256_text(carrier["text"]),
            "injected_carrier_text_sha256": sha256_text(
                injected_carrier_text
            ),
            "injected_carrier_character_count": len(injected_carrier_text),
            "injection_payload_sha256": sha256_text(injection_payload),
            "payload_offset_in_delta": payload_offset_in_delta,
            "payload_length": len(injection_payload),
            "selected_carrier_chunk_id": carrier_chunk["chunk_id"],
            "insertion_offset_in_chunk": (
                insertion_offset - carrier_chunk["source_char_start"]
            ),
            "ranking_used_injection_text": False,
            "carrier_chunk_retention": {
                "policy": carrier_chunk_retention_policy,
                "selection_origin": retention_origin,
                "selected_from_clean_text": True,
                "replaced_chunk_ids": [
                    candidate["chunk_id"]
                    for candidate in replaced_candidates
                ],
            },
        },
    }


def validate_retrieval_plan(
    plan: dict[str, Any],
    documents: list[dict[str, Any]],
    *,
    injection_payload: str,
) -> None:
    """Validate plan provenance and all source slices without a tokenizer."""

    if plan.get("schema_version") != RETRIEVAL_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported retrieval plan schema_version")
    if plan.get("canonical_ranking_treatment") != "clean":
        raise ValueError("retrieval ranking must be constructed from clean text")
    if plan.get("injection_mapping", {}).get("ranking_used_injection_text") is not False:
        raise ValueError("retrieval plan does not prove treatment-blind ranking")

    documents_by_id = {document["doc_id"]: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("source document IDs are not unique")
    sources = plan.get("source_documents")
    if not isinstance(sources, list) or {
        source.get("doc_id") for source in sources
    } != set(documents_by_id):
        raise ValueError("retrieval plan source documents do not match the registry")

    eligibility = plan.get("evidence_eligibility")
    excluded_page_reasons: dict[str, dict[int, str]] = defaultdict(dict)
    if eligibility is not None:
        if (
            eligibility.get("determined_from")
            != "clean_source_section_type"
            or eligibility.get("all_pages_remain_indexed") is not True
        ):
            raise ValueError("retrieval evidence eligibility is not clean-audited")
        raw_exclusions = eligibility.get("non_evidence_pages", {})
        if not isinstance(raw_exclusions, dict):
            raise ValueError("non-evidence page annotations must be an object")
        for doc_id, entries in raw_exclusions.items():
            if doc_id not in documents_by_id or not isinstance(entries, list):
                raise ValueError("non-evidence page annotations are invalid")
            for entry in entries:
                page_number = entry.get("page_number")
                reason = entry.get("reason")
                if (
                    not isinstance(page_number, int)
                    or isinstance(page_number, bool)
                    or page_number in excluded_page_reasons[doc_id]
                    or not isinstance(reason, str)
                    or not reason.strip()
                ):
                    raise ValueError("non-evidence page annotation is invalid")
                excluded_page_reasons[doc_id][page_number] = reason

    candidates = plan.get("candidate_chunks")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("retrieval plan has no candidate chunks")
    candidates_by_id: dict[str, dict[str, Any]] = {}
    pages_by_document: dict[str, set[int]] = defaultdict(set)
    for candidate in candidates:
        chunk_id = candidate.get("chunk_id")
        doc_id = candidate.get("doc_id")
        if not isinstance(chunk_id, str) or chunk_id in candidates_by_id:
            raise ValueError("retrieval candidate IDs are missing or duplicated")
        if doc_id not in documents_by_id:
            raise ValueError(f"{chunk_id} references an unknown source document")
        source_text = documents_by_id[doc_id]["text"]
        start = candidate.get("source_char_start")
        end = candidate.get("source_char_end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(source_text)
        ):
            raise ValueError(f"{chunk_id} has invalid source character offsets")
        chunk_text = source_text[start:end]
        if sha256_text(chunk_text) != candidate.get("text_sha256"):
            raise ValueError(f"{chunk_id} no longer matches its source text")
        if (
            not isinstance(candidate.get("token_count"), int)
            or candidate["token_count"] < 1
        ):
            raise ValueError(f"{chunk_id} has an invalid token count")
        page_number = candidate.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError(f"{chunk_id} has an invalid page number")
        expected_eligible = (
            page_number not in excluded_page_reasons.get(doc_id, {})
        )
        actual_eligible = candidate.get("evidence_eligible", True)
        if actual_eligible is not expected_eligible:
            raise ValueError(f"{chunk_id} has inconsistent evidence eligibility")
        if actual_eligible:
            if (
                not isinstance(candidate.get("bm25_rank"), int)
                or candidate["bm25_rank"] < 1
                or not isinstance(candidate.get("bm25_score"), (int, float))
                or candidate.get("exclusion_reason") is not None
            ):
                raise ValueError(f"{chunk_id} has invalid BM25 metadata")
        elif (
            candidate.get("bm25_rank") is not None
            or candidate.get("bm25_score") is not None
            or candidate.get("exclusion_reason")
            != excluded_page_reasons[doc_id][page_number]
        ):
            raise ValueError(f"{chunk_id} was not cleanly excluded from ranking")
        pages_by_document[doc_id].add(page_number)
        candidates_by_id[chunk_id] = candidate

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("evidence_eligible", True)
    ]
    ranks = sorted(candidate["bm25_rank"] for candidate in eligible_candidates)
    if ranks != list(range(1, len(eligible_candidates) + 1)):
        raise ValueError("eligible BM25 ranks are not unique and contiguous")

    for source in sources:
        doc_id = source["doc_id"]
        document = documents_by_id[doc_id]
        if sha256_text(document["text"]) != source.get("text_sha256"):
            raise ValueError(f"{doc_id} source hash does not match the plan")
        if len(document["text"]) != source.get("character_count"):
            raise ValueError(f"{doc_id} source character count changed")
        page_count = len(_page_spans(document["text"]))
        if page_count != source.get("page_count"):
            raise ValueError(f"{doc_id} source page count changed")
        expected_indexed_pages = [
            page["page_number"]
            for page in _page_spans(document["text"])
            if page["text"].strip()
        ]
        if source.get("indexed_page_numbers") != expected_indexed_pages:
            raise ValueError(f"{doc_id} does not index every non-empty source page")
        if sorted(pages_by_document[doc_id]) != source.get(
            "indexed_page_numbers"
        ):
            raise ValueError(f"{doc_id} does not have complete indexed page coverage")
        if sum(
            candidate["doc_id"] == doc_id for candidate in candidates
        ) != source.get("candidate_chunk_count"):
            raise ValueError(f"{doc_id} candidate count does not match the plan")
        expected_excluded_pages = sorted(
            excluded_page_reasons.get(doc_id, {})
        )
        if any(
            page_number not in expected_indexed_pages
            for page_number in expected_excluded_pages
        ):
            raise ValueError(
                f"{doc_id} has a non-evidence annotation outside the PDF"
            )
        expected_eligible_pages = [
            page_number
            for page_number in expected_indexed_pages
            if page_number not in expected_excluded_pages
        ]
        if eligibility is not None and (
            source.get("non_evidence_page_numbers")
            != expected_excluded_pages
            or source.get("evidence_eligible_page_numbers")
            != expected_eligible_pages
            or source.get("eligible_candidate_chunk_count")
            != sum(
                candidate["doc_id"] == doc_id
                and candidate.get("evidence_eligible") is True
                for candidate in candidates
            )
            or source.get("excluded_candidate_chunk_count")
            != sum(
                candidate["doc_id"] == doc_id
                and candidate.get("evidence_eligible") is False
                for candidate in candidates
            )
        ):
            raise ValueError(f"{doc_id} evidence eligibility summary changed")

    source_pdf_verification = plan.get("source_pdf_verification")
    if source_pdf_verification is not None:
        status = source_pdf_verification.get("status")
        if status not in {"verified", "not_performed"}:
            raise ValueError("source PDF verification has an invalid status")
        if status == "verified":
            verified_documents = source_pdf_verification.get("documents")
            if not isinstance(verified_documents, list) or {
                item.get("doc_id") for item in verified_documents
            } != set(documents_by_id):
                raise ValueError(
                    "source PDF verification does not cover every clean document"
                )
            sources_by_id = {
                source["doc_id"]: source for source in sources
            }
            for item in verified_documents:
                doc_id = item["doc_id"]
                source = sources_by_id[doc_id]
                expected_filename = (
                    source.get("clean_source_pdf")
                    if source["role"] == "injection_carrier"
                    else source.get("source_pdf")
                )
                if (
                    item.get("filename") != expected_filename
                    or not _SHA256.fullmatch(str(item.get("pdf_sha256", "")))
                    or item.get("extracted_text_sha256")
                    != source["text_sha256"]
                    or item.get("matches_indexed_text_fixture") is not True
                ):
                    raise ValueError(
                        f"{doc_id} source PDF verification is inconsistent"
                    )

    selection = plan.get("selection", {})
    selected_ids = selection.get("render_order_chunk_ids")
    ranked_ids = selection.get("ranked_chunk_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or set(selected_ids) != set(ranked_ids or [])
        or any(chunk_id not in candidates_by_id for chunk_id in selected_ids)
    ):
        raise ValueError("retrieval plan selection is invalid")
    selected = [candidates_by_id[chunk_id] for chunk_id in selected_ids]
    if len(selected) != selection.get("selected_chunk_count"):
        raise ValueError("selected chunk count does not match the plan")
    if len(candidates) != selection.get("candidate_chunk_count"):
        raise ValueError("candidate chunk count does not match the plan")
    if eligibility is not None and (
        len(eligible_candidates)
        != selection.get("eligible_candidate_chunk_count")
        or len(candidates) - len(eligible_candidates)
        != selection.get("excluded_candidate_chunk_count")
    ):
        raise ValueError("eligible candidate counts do not match the plan")
    if any(
        candidate.get("evidence_eligible", True) is not True
        for candidate in selected
    ):
        raise ValueError("retrieval selected a non-evidence chunk")
    if any(
        candidate.get("selected") is not (
            candidate["chunk_id"] in set(selected_ids)
        )
        for candidate in candidates
    ):
        raise ValueError("retrieval candidate selection flags are inconsistent")
    selected_tokens = sum(candidate["token_count"] for candidate in selected)
    if selected_tokens != selection.get("selected_token_count"):
        raise ValueError("selected token count does not match the plan")
    if selected_tokens > plan.get("budget", {}).get(
        "document_token_budget", -1
    ):
        raise ValueError("retrieval selection exceeds its document token budget")
    if {candidate["doc_id"] for candidate in selected} != set(documents_by_id):
        raise ValueError("retrieval selection omits a source document")
    document_budgets = plan.get("budget", {}).get(
        "document_token_budgets"
    )
    if document_budgets is not None:
        if (
            not isinstance(document_budgets, dict)
            or set(document_budgets) != set(documents_by_id)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
                for value in document_budgets.values()
            )
            or sum(document_budgets.values())
            > plan.get("budget", {}).get("document_token_budget", -1)
        ):
            raise ValueError("per-document token budgets are incomplete")
        for doc_id, budget in document_budgets.items():
            document_tokens = sum(
                candidate["token_count"]
                for candidate in selected
                if candidate["doc_id"] == doc_id
            )
            if (
                not isinstance(budget, int)
                or isinstance(budget, bool)
                or budget < 1
                or document_tokens > budget
            ):
                raise ValueError(f"{doc_id} exceeds its token budget")

    mapping = plan.get("injection_mapping", {})
    carrier_doc_id = mapping.get("carrier_doc_id")
    carrier_document = documents_by_id.get(carrier_doc_id)
    if (
        carrier_document is None
        or carrier_document.get("role") != "injection_carrier"
    ):
        raise ValueError("retrieval injection mapping has the wrong carrier")
    carrier_chunk_id = mapping.get("selected_carrier_chunk_id")
    if carrier_chunk_id not in selected_ids:
        raise ValueError("retrieval selection omitted the injection-bearing chunk")
    carrier_chunk = candidates_by_id[carrier_chunk_id]
    baseline_selected = _select_candidates(
        eligible_candidates,
        [source["doc_id"] for source in sources],
        token_budget=plan["budget"]["document_token_budget"],
        document_token_budgets=document_budgets,
    )
    baseline_ids = {
        candidate["chunk_id"] for candidate in baseline_selected
    }
    retention = mapping.get("carrier_chunk_retention")
    if retention is None:
        if (
            carrier_chunk_id not in baseline_ids
            or set(selected_ids) != baseline_ids
        ):
            raise ValueError(
                "legacy retrieval plans must retain the carrier chunk "
                "naturally"
            )
    else:
        policy = retention.get("policy")
        origin = retention.get("selection_origin")
        replaced_ids = retention.get("replaced_chunk_ids")
        if (
            policy not in CARRIER_CHUNK_RETENTION_POLICIES
            or plan.get("ranking", {}).get(
                "carrier_chunk_retention_policy"
            ) != policy
            or origin not in {
                "natural_clean_rank",
                "controlled_clean_anchor",
            }
            or retention.get("selected_from_clean_text") is not True
            or not isinstance(replaced_ids, list)
            or len(replaced_ids) != len(set(replaced_ids))
            or any(chunk_id not in candidates_by_id for chunk_id in replaced_ids)
        ):
            raise ValueError(
                "retrieval carrier-chunk retention metadata is invalid"
            )
        if origin == "natural_clean_rank":
            if (
                carrier_chunk_id not in baseline_ids
                or replaced_ids
                or set(selected_ids) != baseline_ids
            ):
                raise ValueError(
                    "natural carrier retention does not match the clean "
                    "ranking"
                )
        else:
            if (
                policy != "require_clean_anchor"
                or carrier_chunk_id in baseline_ids
            ):
                raise ValueError(
                    "controlled carrier retention is inconsistent with the "
                    "clean ranking"
                )
            expected_selected, expected_removed = _retain_required_candidate(
                baseline_selected,
                carrier_chunk,
                token_budget=plan["budget"]["document_token_budget"],
                document_token_budgets=document_budgets,
            )
            if (
                set(selected_ids)
                != {
                    candidate["chunk_id"]
                    for candidate in expected_selected
                }
                or replaced_ids
                != [
                    candidate["chunk_id"]
                    for candidate in expected_removed
                ]
            ):
                raise ValueError(
                    "controlled carrier retention does not match the "
                    "clean-anchor replacement rule"
                )
    insertion_offset = mapping.get("source_char_offset")
    relative_offset = mapping.get("insertion_offset_in_chunk")
    if (
        not isinstance(insertion_offset, int)
        or not isinstance(relative_offset, int)
        or insertion_offset - carrier_chunk["source_char_start"] != relative_offset
        or not 0 <= relative_offset <= (
            carrier_chunk["source_char_end"] - carrier_chunk["source_char_start"]
        )
    ):
        raise ValueError("retrieval injection offset is inconsistent")
    delta = mapping.get("insertion_delta")
    if (
        not isinstance(delta, str)
        or sha256_text(delta) != mapping.get("insertion_delta_sha256")
        or sha256_text(carrier_document["text"])
        != mapping.get("clean_carrier_text_sha256")
        or sha256_text(injection_payload)
        != mapping.get("injection_payload_sha256")
        or delta.count(injection_payload) != 1
        or mapping.get("payload_offset_in_delta") != delta.index(injection_payload)
        or mapping.get("payload_length") != len(injection_payload)
    ):
        raise ValueError("retrieval plan does not preserve the exact injection delta")
    reconstructed_injected = (
        carrier_document["text"][:insertion_offset]
        + delta
        + carrier_document["text"][insertion_offset:]
    )
    if (
        sha256_text(reconstructed_injected)
        != mapping.get("injected_carrier_text_sha256")
        or len(reconstructed_injected)
        != mapping.get("injected_carrier_character_count")
    ):
        raise ValueError("retrieval plan does not reconstruct the injected carrier")
    if (
        source_pdf_verification is not None
        and source_pdf_verification.get("status") == "verified"
    ):
        injected_verification = source_pdf_verification.get(
            "injected_carrier",
            {},
        )
        carrier_source = next(
            source
            for source in sources
            if source["doc_id"] == carrier_doc_id
        )
        if (
            injected_verification.get("doc_id") != carrier_doc_id
            or injected_verification.get("filename")
            != carrier_source.get("injected_source_pdf")
            or not _SHA256.fullmatch(
                str(injected_verification.get("pdf_sha256", ""))
            )
            or injected_verification.get("extracted_text_sha256")
            != mapping.get("injected_carrier_text_sha256")
        ):
            raise ValueError("injected source PDF verification is inconsistent")


def load_retrieval_plan(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict):
        raise ValueError("retrieval plan must be a JSON object")
    return plan


def materialize_retrieval(
    *,
    plan: dict[str, Any],
    documents: list[dict[str, Any]],
    treatment: str,
    injection_payload: str,
) -> tuple[list[dict[str, Any]], list[int] | None, dict[str, Any]]:
    """Materialize the fixed selected views and apply the exact injection delta."""

    if treatment not in {"clean", "injected"}:
        raise ValueError("treatment must be clean or injected")
    validate_retrieval_plan(
        plan,
        documents,
        injection_payload=injection_payload,
    )
    candidates_by_id = {
        candidate["chunk_id"]: candidate
        for candidate in plan["candidate_chunks"]
    }
    selected_ids = plan["selection"]["render_order_chunk_ids"]
    selected_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id in selected_ids:
        selected_by_document[candidates_by_id[chunk_id]["doc_id"]].append(
            candidates_by_id[chunk_id]
        )

    mapping = plan["injection_mapping"]
    carrier_chunk_id = mapping["selected_carrier_chunk_id"]
    payload_span: list[int] | None = None
    materialized = []
    trace_chunks = []
    for document in documents:
        parts: list[str] = []
        document_cursor = 0
        document_chunk_ids = []
        page_numbers = []
        selected_token_count = 0
        for candidate in selected_by_document[document["doc_id"]]:
            source_chunk = document["text"][
                candidate["source_char_start"]:candidate["source_char_end"]
            ]
            chunk_text = source_chunk
            payload_offset_in_chunk = None
            if treatment == "injected" and candidate["chunk_id"] == carrier_chunk_id:
                insertion_offset = mapping["insertion_offset_in_chunk"]
                delta = mapping["insertion_delta"]
                chunk_text = (
                    source_chunk[:insertion_offset]
                    + delta
                    + source_chunk[insertion_offset:]
                )
                payload_offset_in_chunk = (
                    insertion_offset + mapping["payload_offset_in_delta"]
                )

            header = (
                f"[Retrieved chunk {candidate['chunk_id']} | "
                f"source page {candidate['page_number']} | "
                f"BM25 rank {candidate['bm25_rank']}]\n"
            )
            separator = "\n\n" if parts else ""
            part_start = document_cursor + len(separator)
            parts.append(header + chunk_text)
            if payload_offset_in_chunk is not None:
                start = part_start + len(header) + payload_offset_in_chunk
                payload_span = [start, start + len(injection_payload)]
            document_cursor = (
                part_start + len(header) + len(chunk_text)
            )
            document_chunk_ids.append(candidate["chunk_id"])
            page_numbers.append(candidate["page_number"])
            selected_token_count += candidate["token_count"]
            trace_chunks.append({
                "chunk_id": candidate["chunk_id"],
                "doc_id": candidate["doc_id"],
                "page_number": candidate["page_number"],
                "bm25_rank": candidate["bm25_rank"],
                "bm25_score": candidate["bm25_score"],
                "token_count": candidate["token_count"],
            })

        output_document = copy.deepcopy(document)
        output_document["text"] = "\n\n".join(parts)
        output_document["retrieval"] = {
            "profile_id": plan["profile_id"],
            "selected_chunk_ids": document_chunk_ids,
            "source_page_numbers": page_numbers,
            "selected_token_count": selected_token_count,
            "source_text_sha256": sha256_text(document["text"]),
        }
        materialized.append(output_document)

    if treatment == "injected" and payload_span is None:
        raise AssertionError("the injection payload was not materialized")
    if treatment == "clean" and payload_span is not None:
        raise AssertionError("the clean retrieval contains an injection span")
    trace = {
        "schema_version": RETRIEVAL_PLAN_SCHEMA_VERSION,
        "profile_id": plan["profile_id"],
        "plan_sha256": canonical_plan_sha256(plan),
        "query": plan["query"],
        "canonical_ranking_treatment": plan["canonical_ranking_treatment"],
        "ranking_used_injection_text": False,
        "candidate_chunk_count": plan["selection"]["candidate_chunk_count"],
        "eligible_candidate_chunk_count": plan["selection"].get(
            "eligible_candidate_chunk_count",
            plan["selection"]["candidate_chunk_count"],
        ),
        "excluded_candidate_chunk_count": plan["selection"].get(
            "excluded_candidate_chunk_count",
            0,
        ),
        "selected_chunk_count": plan["selection"]["selected_chunk_count"],
        "selected_token_count": plan["selection"]["selected_token_count"],
        "document_token_budget": plan["budget"]["document_token_budget"],
        "document_token_budgets": copy.deepcopy(
            plan["budget"].get("document_token_budgets")
        ),
        "context_window_tokens": plan["budget"]["context_window_tokens"],
        "max_new_tokens": plan["budget"]["max_new_tokens"],
        "non_document_reserve_tokens": plan["budget"][
            "non_document_reserve_tokens"
        ],
        "selected_chunk_ids": list(selected_ids),
        "selected_chunks": trace_chunks,
        "source_documents": copy.deepcopy(plan["source_documents"]),
        "evidence_eligibility": copy.deepcopy(
            plan.get("evidence_eligibility")
        ),
        "source_pdf_verification": copy.deepcopy(
            plan.get("source_pdf_verification")
        ),
        "injection_bearing_chunk_id": carrier_chunk_id,
        "carrier_chunk_retention": copy.deepcopy(
            mapping.get("carrier_chunk_retention")
        ),
    }
    return materialized, payload_span, trace
