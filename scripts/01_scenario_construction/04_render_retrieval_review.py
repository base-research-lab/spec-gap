#!/usr/bin/env python3
"""Render a human-readable review packet for a Scenario 1 retrieval plan."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.scenario1 import generator  # noqa: E402
from src.scenario1.retrieval import load_retrieval_plan  # noqa: E402


DEFAULT_REGISTRY = (
    generator.INPUTS / "fellow_packages" / "aihc" / "domain_config.json"
)
DEFAULT_OUTPUT = (
    generator.INPUTS
    / "fellow_packages"
    / "aihc"
    / "retrieval"
    / "aihc_retrieval_review.html"
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _highlight(text: str, payload: str) -> str:
    if payload not in text:
        return _escape(text)
    before, after = text.split(payload, 1)
    return (
        _escape(before)
        + '<mark class="injection">'
        + _escape(payload)
        + "</mark>"
        + _escape(after)
    )


def _selection_rows(
    *,
    documents: list[dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    candidates = {
        item["chunk_id"]: item for item in plan["candidate_chunks"]
    }
    selected_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk_id in plan["selection"]["render_order_chunk_ids"]:
        selected_by_document[candidates[chunk_id]["doc_id"]].append(
            candidates[chunk_id]
        )
    source_by_id = {
        source["doc_id"]: source for source in plan["source_documents"]
    }
    rows = []
    for document in documents:
        selected = selected_by_document[document["doc_id"]]
        pages = sorted({item["page_number"] for item in selected})
        tokens = sum(item["token_count"] for item in selected)
        source = source_by_id[document["doc_id"]]
        document_budget = plan["budget"].get(
            "document_token_budgets"
        )
        token_cap = (
            document_budget.get(document["doc_id"])
            if isinstance(document_budget, dict)
            else None
        )
        role = (
            '<span class="badge carrier">carrier</span>'
            if document["role"] == "injection_carrier"
            else '<span class="badge benign">benign</span>'
        )
        rows.append(
            "<tr>"
            f"<td><strong>{_escape(document['doc_id'])}</strong><br>"
            f"<span class=\"muted\">{_escape(document['title'])}</span></td>"
            f"<td>{role}</td>"
            f"<td>{len(selected)}</td>"
            f"<td>{_escape(', '.join(map(str, pages)))}</td>"
            f"<td>{tokens:,}</td>"
            f"<td>{f'{token_cap:,}' if token_cap is not None else 'global'}</td>"
            f"<td>{source['page_count']}</td>"
            "</tr>"
        )
    return "\n".join(rows), selected_by_document


def _eligibility_rows(
    *,
    documents: list[dict[str, Any]],
    plan: dict[str, Any],
) -> str:
    eligibility = plan.get("evidence_eligibility") or {}
    exclusions = eligibility.get("non_evidence_pages", {})
    sources = {
        source["doc_id"]: source for source in plan["source_documents"]
    }
    rows = []
    for document in documents:
        doc_id = document["doc_id"]
        source = sources[doc_id]
        entries = exclusions.get(doc_id, [])
        excluded_pages = [entry["page_number"] for entry in entries]
        reasons = list(dict.fromkeys(entry["reason"] for entry in entries))
        rows.append(
            "<tr>"
            f"<td><strong>{_escape(doc_id)}</strong><br>"
            f"<span class=\"muted\">{_escape(document['title'])}</span></td>"
            f"<td>{source['page_count']}</td>"
            f"<td>{_escape(', '.join(map(str, excluded_pages)) or 'none')}</td>"
            f"<td>{'<br>'.join(_escape(reason) for reason in reasons) or '—'}</td>"
            f"<td>{source.get('eligible_candidate_chunk_count', source['candidate_chunk_count'])}</td>"
            f"<td>{source.get('excluded_candidate_chunk_count', 0)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _pdf_verification_table(plan: dict[str, Any]) -> str:
    verification = plan["source_pdf_verification"]
    rows = []
    for item in verification["documents"]:
        rows.append(
            "<tr>"
            f"<td>{_escape(item['doc_id'])}</td>"
            f"<td>{_escape(item['filename'])}</td>"
            f"<td><code>{_escape(item['pdf_sha256'])}</code></td>"
            '<td><span class="status ok">exact match</span></td>'
            "</tr>"
        )
    injected = verification["injected_carrier"]
    rows.append(
        "<tr>"
        f"<td>{_escape(injected['doc_id'])} (injected twin)</td>"
        f"<td>{_escape(injected['filename'])}</td>"
        f"<td><code>{_escape(injected['pdf_sha256'])}</code></td>"
        '<td><span class="status ok">hash verified</span></td>'
        "</tr>"
    )
    return "\n".join(rows)


def _chunk_sections(
    *,
    documents: list[dict[str, Any]],
    plan: dict[str, Any],
    selected_by_document: dict[str, list[dict[str, Any]]],
    payload: str,
) -> str:
    mapping = plan["injection_mapping"]
    sections = []
    for document in documents:
        chunks = []
        for candidate in selected_by_document[document["doc_id"]]:
            clean_chunk = document["text"][
                candidate["source_char_start"]:candidate["source_char_end"]
            ]
            is_injection_chunk = (
                candidate["chunk_id"]
                == mapping["selected_carrier_chunk_id"]
            )
            injected_note = ""
            injected_preview = ""
            if is_injection_chunk:
                offset = mapping["insertion_offset_in_chunk"]
                injected_chunk = (
                    clean_chunk[:offset]
                    + mapping["insertion_delta"]
                    + clean_chunk[offset:]
                )
                injected_note = (
                    '<span class="badge injection-badge">'
                    "injection-bearing chunk</span>"
                )
                injected_preview = (
                    '<div class="paired compact">'
                    "<section><h4>Clean chunk</h4>"
                    f"<pre>{_escape(clean_chunk)}</pre></section>"
                    "<section><h4>Injected chunk</h4>"
                    f"<pre>{_highlight(injected_chunk, payload)}</pre></section>"
                    "</div>"
                )
            else:
                injected_preview = f"<pre>{_escape(clean_chunk)}</pre>"
            chunks.append(
                "<details class=\"chunk\">"
                "<summary>"
                f"Page {candidate['page_number']} · "
                f"{_escape(candidate['chunk_id'])} · "
                f"rank {candidate['bm25_rank']} · "
                f"{candidate['token_count']:,} tokens "
                f"{injected_note}"
                "</summary>"
                f"<p class=\"muted\">BM25 score "
                f"{candidate['bm25_score']:.4f}; source characters "
                f"{candidate['source_char_start']:,}–"
                f"{candidate['source_char_end']:,}.</p>"
                f"{injected_preview}"
                "</details>"
            )
        sections.append(
            "<section class=\"document-section\">"
            f"<h3>{_escape(document['doc_id'])}: "
            f"{_escape(document['title'])}</h3>"
            f"<p>{len(chunks)} selected chunks. Open each row to inspect "
            "the exact source text.</p>"
            + "\n".join(chunks)
            + "</section>"
        )
    return "\n".join(sections)


def _model_view_sections(
    *,
    clean_record: dict[str, Any],
    injected_record: dict[str, Any],
    payload: str,
) -> str:
    blocks = []
    for treatment, record in (
        ("Clean", clean_record),
        ("Injected", injected_record),
    ):
        for document in record["document_set"]["documents"]:
            text = document["text"]
            rendered = (
                _highlight(text, payload)
                if treatment == "Injected"
                else _escape(text)
            )
            blocks.append(
                "<details class=\"model-view\">"
                f"<summary>{treatment} · {_escape(document['doc_id'])} · "
                f"{len(text):,} characters</summary>"
                f"<pre>{rendered}</pre>"
                "</details>"
            )
    return "\n".join(blocks)


def build_review(
    registry: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    documents = generator.load_documents(registry)
    clean_record = generator.build_record(registry, "2-hop", "clean")
    injected_record = generator.build_record(
        registry,
        "2-hop",
        "injected",
    )
    payload = registry["injection"]["wordings"][
        registry["assigned_wording"]
    ]
    clean_carrier = next(
        document
        for document in clean_record["document_set"]["documents"]
        if document["role"] == "injection_carrier"
    )
    injected_carrier = next(
        document
        for document in injected_record["document_set"]["documents"]
        if document["role"] == "injection_carrier"
    )
    mapping = plan["injection_mapping"]
    retention = mapping.get("carrier_chunk_retention") or {
        "policy": "natural_only",
        "selection_origin": "natural_clean_rank",
        "replaced_chunk_ids": [],
    }
    domain_label = registry["domain_id"].replace("_", " ").title()
    payload_span = injected_record["injection"][
        "injection_char_span_in_retrieved_doc"
    ]
    pair_delta = mapping["insertion_delta"]
    pair_offset = payload_span[0] - mapping["payload_offset_in_delta"]
    if (
        injected_carrier["text"][:pair_offset]
        + injected_carrier["text"][pair_offset + len(pair_delta):]
        != clean_carrier["text"]
    ):
        raise AssertionError(
            "removing the recorded PDF delta does not recover the clean view"
        )
    if (
        injected_carrier["text"][
            pair_offset:pair_offset + len(pair_delta)
        ]
        != pair_delta
    ):
        raise AssertionError("rendered pair does not preserve the exact PDF delta")
    if payload in clean_carrier["text"]:
        raise AssertionError("clean preview contains the injection payload")
    if injected_carrier["text"].count(payload) != 1:
        raise AssertionError("injected preview must contain the payload once")

    selection_rows, selected_by_document = _selection_rows(
        documents=documents,
        plan=plan,
    )
    chunk_sections = _chunk_sections(
        documents=documents,
        plan=plan,
        selected_by_document=selected_by_document,
        payload=payload,
    )
    model_views = _model_view_sections(
        clean_record=clean_record,
        injected_record=injected_record,
        payload=payload,
    )
    verification_rows = _pdf_verification_table(plan)
    eligibility_rows = _eligibility_rows(
        documents=documents,
        plan=plan,
    )
    preflight_path = (
        generator.INPUTS
        / registry["retrieval"]["context_preflight_file"]
    )
    preflight = json.loads(preflight_path.read_text())
    minimum_headroom = min(
        case["headroom_tokens"] for case in preflight["cases"]
    )
    source_pages = sum(
        source["page_count"] for source in plan["source_documents"]
    )
    selection = plan["selection"]
    plan_hash = clean_record["retrieval_trace"]["plan_sha256"]
    eligible_count = selection.get(
        "eligible_candidate_chunk_count",
        selection["candidate_chunk_count"],
    )
    excluded_count = selection.get("excluded_candidate_chunk_count", 0)
    selection_balance = " / ".join(
        str(len(selected_by_document[document["doc_id"]]))
        for document in documents
    )
    document_budget = plan["budget"].get("document_token_budgets")
    document_budget_text = (
        ", ".join(
            f"{doc_id}: {budget:,}"
            for doc_id, budget in document_budget.items()
        )
        if isinstance(document_budget, dict)
        else "one global budget"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(domain_label)} full-corpus retrieval review</title>
<style>
:root {{
  --ink: #172033;
  --muted: #626b7c;
  --line: #dce1ea;
  --paper: #ffffff;
  --wash: #f5f7fb;
  --navy: #252a68;
  --teal: #087d70;
  --red: #a83d35;
  --gold: #ffd66b;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--wash);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ max-width: 1320px; margin: 0 auto; padding: 32px 24px 80px; }}
header {{
  color: white;
  background: var(--navy);
  border-radius: 18px;
  padding: 28px 32px;
  margin-bottom: 22px;
}}
h1 {{ margin: 0 0 8px; font-size: 32px; }}
h2 {{ margin-top: 34px; }}
h3 {{ margin: 0 0 8px; }}
h4 {{ margin: 0 0 8px; }}
p {{ max-width: 900px; }}
.subtitle {{ color: #dce2ff; margin: 0; }}
.cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
  margin: 18px 0;
}}
.card, .panel {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
}}
.card strong {{ display: block; font-size: 25px; color: var(--navy); }}
.card span, .muted {{ color: var(--muted); }}
.callout {{
  border-left: 5px solid var(--teal);
  background: #e9f7f4;
  padding: 14px 18px;
  border-radius: 8px;
  margin: 18px 0;
}}
.warning {{ border-left-color: #d38d00; background: #fff8df; }}
.checklist li {{ margin: 7px 0; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--paper);
  border: 1px solid var(--line);
}}
th, td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid var(--line); }}
th {{ color: var(--navy); background: #eef0fa; }}
code {{ font-size: 12px; overflow-wrap: anywhere; }}
.badge, .status {{
  display: inline-block;
  border-radius: 99px;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 700;
}}
.carrier {{ color: #7b2a25; background: #ffe7e4; }}
.benign {{ color: #17645c; background: #daf2ed; }}
.injection-badge {{ color: #7b2a25; background: #ffd9d5; margin-left: 8px; }}
.ok {{ color: #17645c; background: #daf2ed; }}
.document-section {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 18px;
  margin: 16px 0;
}}
details {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 8px 0;
}}
summary {{
  cursor: pointer;
  padding: 12px 14px;
  font-weight: 650;
}}
details > p, details > pre, details > .paired {{ margin: 0 14px 14px; }}
pre {{
  margin: 0;
  padding: 14px;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f8f9fc;
  border: 1px solid #e6e9ef;
  border-radius: 7px;
  font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.paired {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}}
mark.injection {{
  color: #43130f;
  background: var(--gold);
  outline: 2px solid #e6a900;
  border-radius: 2px;
}}
.meta {{
  display: grid;
  grid-template-columns: 190px 1fr;
  gap: 6px 14px;
}}
.meta dt {{ color: var(--muted); }}
.meta dd {{ margin: 0; overflow-wrap: anywhere; }}
@media (max-width: 900px) {{
  .cards {{ grid-template-columns: repeat(2, 1fr); }}
  .paired {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>{_escape(domain_label)} full-corpus retrieval review</h1>
  <p class="subtitle">Profile {_escape(plan['profile_id'])} · human sign-off
  before a paid Qwen/Modal run</p>
</header>

<div class="cards">
  <div class="card"><strong>{source_pages}</strong><span>PDF pages indexed</span></div>
  <div class="card"><strong>{selection['candidate_chunk_count']}</strong><span>all indexed chunks</span></div>
  <div class="card"><strong>{eligible_count}</strong><span>evidence-eligible chunks</span></div>
  <div class="card"><strong>{excluded_count}</strong><span>non-evidence chunks</span></div>
  <div class="card"><strong>{selection['selected_chunk_count']}</strong><span>selected chunks</span></div>
  <div class="card"><strong>{selection['selected_token_count']:,}</strong><span>selected Qwen tokens</span></div>
  <div class="card"><strong>{minimum_headroom:,}</strong><span>minimum context headroom</span></div>
</div>

<div class="callout">
  <strong>What you are approving:</strong> whether these passages are a fair,
  task-relevant evidence set. Every source page remains indexed and auditable.
  Clean-only section labels prevent the {excluded_count} non-evidence chunks
  from entering BM25 ranking. Worker1 sees only the selected chunks below, and
  the hidden directive did not participate in either eligibility or ranking.
</div>

<section class="panel">
  <h2>Human review checklist</h2>
  <ol class="checklist">
    <li>Do the selected passages adequately support the stated benign task?</li>
    <li>Is the {selection_balance} selected-chunk balance across the three
    documents fair, without one paper dominating?</li>
    <li>Are the listed non-evidence pages correctly limited to references,
    contributor material, licensing, or publication metadata?</li>
    <li>Does the carrier passage remain coherent at its page boundary?</li>
    <li>Is the yellow highlighted directive the only meaningful difference
    between the clean and injected views?</li>
    <li>If all five answers are acceptable, the package can proceed to the
    selected Modal smoke or batch run.</li>
  </ol>
</section>

<h2>Selection summary</h2>
<table>
<thead><tr><th>Document</th><th>Role</th><th>Chunks</th><th>Selected pages</th><th>Tokens</th><th>Token cap</th><th>All pages</th></tr></thead>
<tbody>{selection_rows}</tbody>
</table>

<h2>Evidence-eligibility audit</h2>
<div class="callout">
  <strong>Indexed is not the same as selected.</strong> The pages below remain
  in the complete audit index, but their clean section type makes them
  ineligible for relevance scoring or selection.
</div>
<table>
<thead><tr><th>Document</th><th>All pages</th><th>Non-evidence pages</th><th>Reason</th><th>Eligible chunks</th><th>Excluded chunks</th></tr></thead>
<tbody>{eligibility_rows}</tbody>
</table>

<h2>Retrieval controls</h2>
<dl class="meta panel">
  <dt>Query</dt><dd>{_escape(plan['query'])}</dd>
  <dt>Query construction</dt><dd>{_escape(registry['retrieval']['query_construction'])}</dd>
  <dt>Ranking</dt><dd>BM25 on clean, evidence-eligible source chunks only</dd>
  <dt>Selection policy</dt><dd>{_escape(plan['ranking']['selection_policy'])}</dd>
  <dt>Carrier retention</dt><dd>{_escape(retention['selection_origin'])}
  under policy {_escape(retention['policy'])}; replaced clean-ranked chunks:
  {_escape(', '.join(retention['replaced_chunk_ids']) or 'none')}</dd>
  <dt>Per-document caps</dt><dd>{_escape(document_budget_text)} Qwen tokens</dd>
  <dt>Chunking</dt><dd>{plan['chunking']['chunk_tokens']:,} tokens with
  {plan['chunking']['overlap_tokens']:,}-token overlap; never crosses a PDF page</dd>
  <dt>Plan SHA-256</dt><dd><code>{_escape(plan_hash)}</code></dd>
  <dt>Carrier chunk</dt><dd>{_escape(mapping['selected_carrier_chunk_id'])},
  clean BM25 rank {next(item['bm25_rank'] for item in plan['candidate_chunks'] if item['chunk_id'] == mapping['selected_carrier_chunk_id'])}</dd>
  <dt>Matched-pair check</dt><dd>One exact {len(pair_delta):,}-character insertion
  at rendered carrier offset {pair_offset:,}; payload occurs zero times in clean
  and once in injected.</dd>
</dl>

<h2>Source PDF verification</h2>
<p>Each named PDF was checked against its registry hash. Every clean PDF was
re-extracted with <code>pdftotext -raw</code> and matched its indexed fixture
exactly.</p>
<table>
<thead><tr><th>Document</th><th>PDF</th><th>SHA-256</th><th>Result</th></tr></thead>
<tbody>{verification_rows}</tbody>
</table>

<h2>Selected source passages</h2>
<p>Open the rows below. These are the original clean source slices used by the
retrieval plan. The carrier row also shows its injected twin side by side.</p>
{chunk_sections}

<h2>Exact retrieved-document component sent to Worker1</h2>
<div class="callout warning">
  The document component below is exact. The complete Worker1 prompt will also
  include the task and the planner's generated message, which does not exist
  until the paid run.
</div>
{model_views}

</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = generator.load_registry(args.registry)
    plan_path = generator.INPUTS / registry["retrieval"]["plan_file"]
    plan = load_retrieval_plan(plan_path)
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = build_review(registry, plan)
    output.write_text(rendered)
    print(f"wrote: {output}")
    print(f"bytes: {output.stat().st_size}")
    print(
        "review status: source PDFs verified; matched treatment pair "
        "reconstructed; no model or GPU called"
    )


if __name__ == "__main__":
    main()
