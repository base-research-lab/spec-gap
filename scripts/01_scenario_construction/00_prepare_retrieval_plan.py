#!/usr/bin/env python3
"""Build a treatment-blind full-corpus retrieval plan for Scenario 1.

The injected carrier may be a PDF or an already extracted UTF-8 text file.
PDFs are extracted with ``pdftotext -raw``, matching the clean fixtures.
This script never calls a model or starts Modal compute.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.infrastructure.qwen_modal import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    generation_settings_for_protocol,
)
from src.scenario1.generator import (  # noqa: E402
    INPUTS,
    generation_protocol_id_for_registry,
    load_documents,
    load_registry,
)
from src.scenario1.retrieval import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    DEFAULT_DOCUMENT_TOKEN_BUDGET,
    DEFAULT_NON_DOCUMENT_RESERVE_TOKENS,
    DEFAULT_PROFILE_ID,
    build_retrieval_plan,
    canonical_plan_sha256,
    validate_retrieval_plan,
)


DEFAULT_AIHC_REGISTRY = (
    INPUTS / "fellow_packages" / "aihc" / "domain_config.json"
)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_pdf(path: Path) -> str:
    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError(
            "pdftotext is required to verify source PDFs with -raw"
        )
    with tempfile.TemporaryDirectory(prefix="specgap-retrieval-") as temp_dir:
        extracted = Path(temp_dir) / "extracted.txt"
        subprocess.run(
            [executable, "-raw", str(path), str(extracted)],
            check=True,
        )
        return extracted.read_text()


def _read_injected_carrier(path: Path) -> str:
    return _extract_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text()


def _verify_source_pdfs(
    *,
    registry: dict,
    documents: list[dict],
    source_pdf_root: Path,
    injected_carrier_path: Path,
    injected_carrier_text: str,
) -> dict:
    expected_hashes = registry.get("provenance", {}).get(
        "source_pdf_sha256",
        {},
    )
    slots_by_id = {
        slot["doc_id"]: slot for slot in registry["document_slots"]
    }
    verified_documents = []
    for document in documents:
        slot = slots_by_id[document["doc_id"]]
        filename = (
            slot.get("clean_source_pdf")
            if document["role"] == "injection_carrier"
            else slot.get("source_pdf")
        )
        if not isinstance(filename, str) or not filename:
            raise ValueError(
                f"{document['doc_id']} has no clean source-PDF filename"
            )
        source_pdf = source_pdf_root / filename
        actual_pdf_hash = _sha256_file(source_pdf)
        if actual_pdf_hash != expected_hashes.get(filename):
            raise ValueError(f"{filename} does not match its registry SHA-256")
        extracted_text = _extract_pdf(source_pdf)
        if extracted_text != document["text"]:
            raise ValueError(
                f"{filename} does not exactly reproduce the indexed text fixture"
            )
        verified_documents.append({
            "doc_id": document["doc_id"],
            "filename": filename,
            "pdf_sha256": actual_pdf_hash,
            "extracted_text_sha256": hashlib.sha256(
                extracted_text.encode("utf-8")
            ).hexdigest(),
            "matches_indexed_text_fixture": True,
        })

    carrier_slot = next(
        slot
        for slot in registry["document_slots"]
        if slot["role"] == "injection_carrier"
    )
    injected_filename = carrier_slot.get("injected_source_pdf")
    if (
        injected_carrier_path.suffix.lower() != ".pdf"
        or injected_carrier_path.name != injected_filename
    ):
        raise ValueError(
            "source-PDF verification requires the named injected carrier PDF"
        )
    injected_pdf_hash = _sha256_file(injected_carrier_path)
    if injected_pdf_hash != expected_hashes.get(injected_filename):
        raise ValueError(
            f"{injected_filename} does not match its registry SHA-256"
        )
    return {
        "status": "verified",
        "source_folder_name": source_pdf_root.name,
        "extraction_command": "pdftotext -raw",
        "documents": verified_documents,
        "injected_carrier": {
            "doc_id": carrier_slot["doc_id"],
            "filename": injected_filename,
            "pdf_sha256": injected_pdf_hash,
            "extracted_text_sha256": hashlib.sha256(
                injected_carrier_text.encode("utf-8")
            ).hexdigest(),
        },
    }


def _non_evidence_page_map(registry: dict) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    raw = registry.get("retrieval", {}).get("non_evidence_pages", {})
    if not isinstance(raw, dict):
        raise ValueError("registry retrieval.non_evidence_pages must be an object")
    for doc_id, groups in raw.items():
        if not isinstance(groups, list):
            raise ValueError(f"{doc_id} non-evidence annotations must be a list")
        page_reasons: dict[int, str] = {}
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(
                    f"{doc_id} non-evidence annotation must be an object"
                )
            pages = group.get("pages")
            reason = group.get("reason")
            if (
                not isinstance(pages, list)
                or not pages
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError(f"{doc_id} has an invalid non-evidence annotation")
            for page_number in pages:
                if (
                    not isinstance(page_number, int)
                    or isinstance(page_number, bool)
                    or page_number < 1
                ):
                    raise ValueError(
                        f"{doc_id} has an invalid non-evidence page number"
                    )
                if page_number in page_reasons:
                    raise ValueError(
                        f"{doc_id} page {page_number} was excluded twice"
                    )
                page_reasons[page_number] = reason
        result[doc_id] = page_reasons
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index every clean PDF page and select a fixed set of clean-ranked "
            "chunks for both Scenario 1 treatments."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_AIHC_REGISTRY,
    )
    parser.add_argument(
        "--injected-carrier",
        type=Path,
        required=True,
        help="Injected carrier PDF or its pdftotext -raw UTF-8 extraction.",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        required=True,
        help="tokenizer.json from the exact pinned Qwen model revision.",
    )
    parser.add_argument(
        "--source-pdf-root",
        type=Path,
        help=(
            "Folder containing the named clean and injected PDFs. When set, "
            "all PDF hashes and clean pdftotext -raw fixtures are verified."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Defaults to retrieval.plan_file from the selected registry.",
    )
    parser.add_argument(
        "--profile-id",
        help="Defaults to retrieval.profile_id from the selected registry.",
    )
    parser.add_argument("--query")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
    )
    parser.add_argument(
        "--document-token-budget",
        type=int,
        default=DEFAULT_DOCUMENT_TOKEN_BUDGET,
    )
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "Install the explicit retrieval dependency first: "
            "python -m pip install 'tokenizers>=0.20,<1'"
        ) from exc

    registry = load_registry(args.registry)
    generation_protocol_id = generation_protocol_id_for_registry(registry)
    generation_settings = generation_settings_for_protocol(
        generation_protocol_id
    )
    retrieval_config = registry.get("retrieval", {})
    profile_id = (
        args.profile_id
        or retrieval_config.get("profile_id")
        or DEFAULT_PROFILE_ID
    )
    configured_plan_file = retrieval_config.get("plan_file")
    if args.out is not None:
        output_path = args.out.resolve()
    elif isinstance(configured_plan_file, str) and configured_plan_file:
        output_path = (INPUTS / configured_plan_file).resolve()
    else:
        raise ValueError("set --out or registry retrieval.plan_file")
    documents = load_documents(registry)
    injected_carrier_text = _read_injected_carrier(
        args.injected_carrier.resolve()
    )
    source_pdf_verification = (
        _verify_source_pdfs(
            registry=registry,
            documents=documents,
            source_pdf_root=args.source_pdf_root.resolve(),
            injected_carrier_path=args.injected_carrier.resolve(),
            injected_carrier_text=injected_carrier_text,
        )
        if args.source_pdf_root is not None
        else {
            "status": "not_performed",
            "reason": "--source-pdf-root was not supplied",
        }
    )
    tokenizer_path = args.tokenizer_json.resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    injection_payload = registry["injection"]["wordings"][
        registry["assigned_wording"]
    ]
    plan = build_retrieval_plan(
        documents=documents,
        injected_carrier_text=injected_carrier_text,
        injection_payload=injection_payload,
        query=(
            args.query
            or retrieval_config.get("query")
            or registry["task"]["user_task"]
        ),
        tokenizer=tokenizer,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        tokenizer_sha256=_sha256_file(tokenizer_path),
        profile_id=profile_id,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        document_token_budget=args.document_token_budget,
        document_token_budgets=copy.deepcopy(
            retrieval_config.get("document_token_budgets")
        ),
        non_evidence_pages=_non_evidence_page_map(registry),
        carrier_chunk_retention_policy=retrieval_config.get(
            "carrier_chunk_retention_policy",
            "natural_only",
        ),
        context_window_tokens=DEFAULT_CONTEXT_WINDOW_TOKENS,
        max_new_tokens=generation_settings["max_new_tokens"],
        non_document_reserve_tokens=DEFAULT_NON_DOCUMENT_RESERVE_TOKENS,
    )
    plan["source_pdf_verification"] = source_pdf_verification
    validate_retrieval_plan(
        plan,
        documents,
        injection_payload=injection_payload,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote: {output_path}")
    print(
        "generation protocol: "
        f"{generation_protocol_id} "
        f"({generation_settings['max_new_tokens']} max new tokens)"
    )
    print(f"plan sha256: {canonical_plan_sha256(plan)}")
    print(
        "indexed: "
        f"{sum(item['page_count'] for item in plan['source_documents'])} pages, "
        f"{plan['selection']['candidate_chunk_count']} candidate chunks"
    )
    print(
        "selected: "
        f"{plan['selection']['selected_chunk_count']} chunks, "
        f"{plan['selection']['selected_token_count']} model tokens "
        f"(budget {plan['budget']['document_token_budget']})"
    )
    retention = plan["injection_mapping"]["carrier_chunk_retention"]
    print(
        "injection-bearing chunk selected from clean text: "
        f"{plan['injection_mapping']['selected_carrier_chunk_id']} "
        f"({retention['selection_origin']}; policy={retention['policy']})"
    )
    print(
        "source PDF verification: "
        f"{plan['source_pdf_verification']['status']}"
    )


if __name__ == "__main__":
    main()
