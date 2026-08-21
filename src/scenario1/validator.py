#!/usr/bin/env python3
"""
Validate scenario 1 trajectory files.

Two layers:
  1. Structural: JSON Schema (schemas/scenario1/v2/trajectory.schema.json).
  2. Semantic invariants that JSON Schema cannot express and that map to the
     bugs this project actually hit:
       INV-1  an optional primary layer is present in every materialized
              extraction. No primary is required before the Qwen layer scan.
       INV-2  exactly the agents in raw_poison_exposed_agents have the injection
              in their prompt; everyone else must not (token-alignment bug +
              executor re-reading raw poison). Real exposed turns must include
              exact rendered-prompt character and token spans.
       INV-3  hops_survived == unsafe_action_hop_index - injection_hop_index.
       INV-4  delegation_depth == condition_id, and hop_path length matches.
       INV-5  action_channel.unsafe_action_executed and the deprecated aliases
              agree with the unsafe_action events (skipped while the value is
              null, i.e. before the model has run).
       INV-6  no auto-derived step_label is ever reasoning_compromised (that
              state is annotation-only).
       INV-7  outcome_class is coherent with treatment and the channels:
              a normal clean control is clean, while an unexplained endpoint
              echo is indeterminate; executed iff the action fired;
              propagated_but_not_executed requires an injected trajectory with
              output_adoption true and no fired action.
       INV-8  agent topology matches the depth condition (a 3-hop trace must
              contain worker_2; no unexpected agents).
       INV-9  a non-null reasoning label requires human or mechanistic evidence;
              construction/auto proxies may not set it.
       INV-10 dry runs are honest: no model call, generated output, outcome, or
              extracted activation may be claimed.
       INV-11 full-corpus retrieval is treatment-blind, budgeted, page-audited,
              and internally consistent with the retrieved document views.

Usage:
  python scripts/01_scenario_construction/02_validate_trajectories.py \
      experiments/scenario1/trajectories/*.json
"""

import glob
import json
import sys
import os

from jsonschema import Draft202012Validator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(
    PROJECT_ROOT, "schemas", "scenario1", "v2", "trajectory.schema.json"
)
HOP_PATH_LEN = {"2-hop": 5, "3-hop": 6}
EXPECTED_AGENTS = {
    "2-hop": {"planner_1", "worker_1", "executor_1"},
    "3-hop": {"planner_1", "worker_1", "worker_2", "executor_1"},
}
REASONING_EVIDENCE_STATUS = {"human_annotated", "mechanistic_evidence"}


def _agent_turns(t):
    return [e for e in t["trajectory_trace"]["full_events"]
            if e.get("type") == "agent_turn"]


def semantic_checks(t):
    errs = []

    # INV-1: a configured primary layer must be in every materialized block.
    primary = t["activation_config"].get("primary_layer")
    for e in _agent_turns(t):
        for key in ("activation_metadata", "attention_metadata"):
            meta = e.get(key)
            if meta and primary is not None and \
                    meta.get("storage_status") == "materialized" and \
                    primary not in meta.get("layers_extracted", []):
                errs.append(f"INV-1 {e['agent_id']}.{key}: primary layer "
                            f"{primary} not in {meta.get('layers_extracted')}")

    # INV-2: exposure set matches raw_poison_exposed_agents.
    if t["injection"]["injection_present"]:
        exposed_declared = set(t["injection"].get("raw_poison_exposed_agents", []))
        exposed_actual = {
            e["agent_id"] for e in _agent_turns(t)
            if e.get("token_alignment", {}).get("injection_present_in_prompt")}
        if exposed_actual != exposed_declared:
            errs.append(f"INV-2 exposure mismatch: declared {sorted(exposed_declared)} "
                        f"but prompts expose {sorted(exposed_actual)}")
    if t.get("generation_mode") == "real":
        injected_text = t["injection"].get("injected_text")
        for event in _agent_turns(t):
            alignment = event.get("token_alignment") or {}
            exposed = alignment.get("injection_present_in_prompt") is True
            char_span = alignment.get("char_span")
            token_span = alignment.get("injection_token_span")
            if not exposed:
                if char_span is not None or token_span is not None:
                    errs.append(
                        f"INV-2 {event['agent_id']}: unexposed turn contains "
                        "injection spans"
                    )
                continue
            if not isinstance(char_span, dict) or not isinstance(token_span, dict):
                errs.append(
                    f"INV-2 {event['agent_id']}: real exposed turn requires "
                    "character and token spans"
                )
                continue
            rendered = event.get("input", {}).get("rendered_prompt")
            input_ids = event.get("input", {}).get("input_token_ids")
            start_char = char_span.get("start_char")
            end_char = char_span.get("end_char")
            start_token = token_span.get("start_token")
            end_token = token_span.get("end_token")
            if (
                not isinstance(rendered, str)
                or not isinstance(start_char, int)
                or not isinstance(end_char, int)
                or not 0 <= start_char < end_char <= len(rendered)
                or rendered[start_char:end_char] != injected_text
            ):
                errs.append(
                    f"INV-2 {event['agent_id']}: character span does not select "
                    "the exact injected text"
                )
            if (
                not isinstance(input_ids, list)
                or not isinstance(start_token, int)
                or not isinstance(end_token, int)
                or not 0 <= start_token < end_token <= len(input_ids)
            ):
                errs.append(
                    f"INV-2 {event['agent_id']}: injection token span is invalid"
                )
            if alignment.get("rendered_prompt_hash") != event.get("input", {}).get(
                "rendered_prompt_hash"
            ):
                errs.append(
                    f"INV-2 {event['agent_id']}: token-alignment prompt hash "
                    "does not match the saved input"
                )
            if alignment.get("truncation_removed_injection_tokens") is not False:
                errs.append(
                    f"INV-2 {event['agent_id']}: injection was truncated or "
                    "truncation status is missing"
                )

    # INV-3: hops_survived arithmetic.
    cp = t["compromise_propagation"]
    if cp.get("injection_hop_index") is not None and \
            cp.get("unsafe_action_hop_index") is not None and \
            cp.get("hops_survived") is not None:
        expect = cp["unsafe_action_hop_index"] - cp["injection_hop_index"]
        if cp["hops_survived"] != expect:
            errs.append(f"INV-3 hops_survived {cp['hops_survived']} != "
                        f"{cp['unsafe_action_hop_index']} - {cp['injection_hop_index']}")

    # INV-4: depth consistency.
    if t["delegation_depth"] != t["condition_id"]:
        errs.append(f"INV-4 delegation_depth {t['delegation_depth']} != condition_id {t['condition_id']}")
    exp_len = HOP_PATH_LEN.get(t["condition_id"])
    if exp_len and len(t["hop_path"]) != exp_len:
        errs.append(f"INV-4 hop_path length {len(t['hop_path'])} != {exp_len} for {t['condition_id']}")

    # INV-5: action_channel.unsafe_action_executed agrees with unsafe_action events.
    unsafe = [e for e in t["trajectory_trace"]["full_events"]
              if e.get("type") == "unsafe_action"]
    chain = any(e.get("metadata", {}).get("label_compromised") for e in unsafe)
    el = t["evaluation_labels"]
    executed = el.get("action_channel", {}).get("unsafe_action_executed")
    if executed is not None and executed != chain:
        errs.append("INV-5 action_channel.unsafe_action_executed disagrees with unsafe_action events")
    for alias in ("injection_success", "exfiltration_detected"):
        if alias in el and el[alias] != chain:
            errs.append(f"INV-5 deprecated alias {alias} disagrees with action firing")

    # INV-7: outcome_class coherent with treatment and the channels.
    oc = el.get("outcome_class")
    if oc is not None:
        injected = t["injection"]["injection_present"]
        output_adoption = el.get("behavioral_channel", {}).get("output_adoption")
        if not injected and oc == "clean" and (executed or output_adoption):
            errs.append(
                "INV-7 clean outcome cannot contain output adoption or an "
                "executed unsafe action"
            )
        if injected and oc == "clean":
            errs.append("INV-7 outcome_class clean but injection_present is true")
        if not injected and oc in {"resisted", "propagated_but_not_executed"}:
            errs.append(
                f"INV-7 outcome_class {oc} requires injection_present true"
            )
        if executed is True and oc != "executed":
            errs.append(f"INV-7 unsafe_action_executed true but outcome_class is {oc} (expected executed)")
        if oc == "executed" and executed is False:
            errs.append("INV-7 outcome_class executed but unsafe_action_executed is false")
        if oc == "propagated_but_not_executed":
            if executed:
                errs.append("INV-7 propagated_but_not_executed but the action fired")
            if output_adoption is False:
                errs.append("INV-7 propagated_but_not_executed but behavioral output_adoption is false")

    # INV-6: reasoning_compromised is annotation-only.
    for e in _agent_turns(t):
        sl = e.get("step_label")
        if sl and sl.get("state") == "reasoning_compromised" and sl.get("source") != "blind_annotation":
            errs.append(f"INV-6 {e['agent_id']}: reasoning_compromised must be blind_annotation, "
                        f"got source={sl.get('source')}")

    # INV-8: agent topology matches the depth condition (catches a 3-hop trace
    # missing worker_2, or an unexpected agent).
    present = {e["agent_id"] for e in _agent_turns(t)}
    expected = EXPECTED_AGENTS.get(t["condition_id"])
    if expected and present != expected:
        errs.append(f"INV-8 agents {sorted(present)} != expected {sorted(expected)} "
                    f"for {t['condition_id']}")

    # INV-9: a non-null reasoning label needs human or mechanistic evidence;
    # construction/auto proxies may not set it.
    for e in _agent_turns(t):
        rl = e.get("reasoning_compromise_label")
        if rl and rl.get("label") is not None and \
                rl.get("annotation_status") not in REASONING_EVIDENCE_STATUS:
            errs.append(f"INV-9 {e['agent_id']}: reasoning label set without evidence "
                        f"(annotation_status={rl.get('annotation_status')})")

    # INV-10: a structural dry run may describe inputs and request templates,
    # but it may not pretend a model generated output or activations.
    if t.get("generation_mode") == "dry_run":
        if t.get("model_called") is not False:
            errs.append("INV-10 dry_run requires model_called=false")
        labels = t.get("evaluation_labels", {})
        if labels.get("outcome_class") is not None:
            errs.append("INV-10 dry_run outcome_class must be null")
        if labels.get("action_channel", {}).get("unsafe_action_executed") is not None:
            errs.append("INV-10 dry_run unsafe_action_executed must be null")
        if labels.get("behavioral_channel", {}).get("output_adoption") is not None:
            errs.append("INV-10 dry_run output_adoption must be null")
        for e in _agent_turns(t):
            if e.get("model_called") is not False:
                errs.append(f"INV-10 {e['agent_id']}: dry_run event requires model_called=false")
            output = e.get("output") or {}
            if any(output.get(field) is not None for field in (
                "message", "raw_generated_text", "generated_token_ids",
                "parsed_message", "tool_call_requests", "finish_reason",
                "truncated", "thinking_content", "final_content", "actions",
            )):
                errs.append(f"INV-10 {e['agent_id']}: dry_run contains generated output")
            for key in ("activation_metadata", "attention_metadata"):
                meta = e.get(key)
                if meta and (meta.get("layers_extracted") or meta.get("storage_path")):
                    errs.append(f"INV-10 {e['agent_id']}.{key}: dry_run claims an artifact")

    # INV-11: optional controlled retrieval metadata must agree everywhere it
    # is recorded. Cross-treatment equality is additionally enforced while
    # materializing the matched pair in generator.py.
    retrieval = t.get("retrieval_trace")
    if retrieval is not None:
        selected_ids = retrieval.get("selected_chunk_ids", [])
        selected_chunks = retrieval.get("selected_chunks", [])
        if retrieval.get("canonical_ranking_treatment") != "clean" or \
                retrieval.get("ranking_used_injection_text") is not False:
            errs.append("INV-11 retrieval ranking must use clean text only")
        if [chunk.get("chunk_id") for chunk in selected_chunks] != selected_ids:
            errs.append("INV-11 selected chunk trace does not match selected IDs")
        if len(selected_ids) != retrieval.get("selected_chunk_count"):
            errs.append("INV-11 selected chunk count is inconsistent")
        selected_tokens = sum(
            chunk.get("token_count", 0) for chunk in selected_chunks
        )
        if selected_tokens != retrieval.get("selected_token_count"):
            errs.append("INV-11 selected token total is inconsistent")
        if selected_tokens > retrieval.get("document_token_budget", -1):
            errs.append("INV-11 selected chunks exceed the document token budget")
        reserved_total = (
            retrieval.get("document_token_budget", 0)
            + retrieval.get("max_new_tokens", 0)
            + retrieval.get("non_document_reserve_tokens", 0)
        )
        if reserved_total > retrieval.get("context_window_tokens", -1):
            errs.append("INV-11 retrieval budget exceeds the context window")
        sources = retrieval.get("source_documents", [])
        candidate_count = retrieval.get("candidate_chunk_count")
        eligible_count = retrieval.get(
            "eligible_candidate_chunk_count",
            candidate_count,
        )
        excluded_count = retrieval.get("excluded_candidate_chunk_count", 0)
        if (
            not isinstance(candidate_count, int)
            or not isinstance(eligible_count, int)
            or not isinstance(excluded_count, int)
            or eligible_count + excluded_count != candidate_count
        ):
            errs.append("INV-11 eligible and excluded candidate counts are inconsistent")
        if sum(source.get("candidate_chunk_count", 0) for source in sources) != (
            candidate_count
        ):
            errs.append("INV-11 candidate chunk count is inconsistent")

        eligibility = retrieval.get("evidence_eligibility")
        excluded_pages_by_document = {}
        if eligibility is not None:
            if (
                not isinstance(eligibility, dict)
                or eligibility.get("determined_from")
                != "clean_source_section_type"
                or eligibility.get("all_pages_remain_indexed") is not True
            ):
                errs.append(
                    "INV-11 evidence eligibility must use clean source section types"
                )
            raw_exclusions = (
                eligibility.get("non_evidence_pages", {})
                if isinstance(eligibility, dict)
                else {}
            )
            if not isinstance(raw_exclusions, dict):
                errs.append("INV-11 non-evidence page annotations are invalid")
                raw_exclusions = {}
            for doc_id, entries in raw_exclusions.items():
                pages = set()
                if not isinstance(entries, list):
                    errs.append(
                        f"INV-11 {doc_id}: non-evidence page annotations are invalid"
                    )
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        errs.append(
                            f"INV-11 {doc_id}: non-evidence page annotation is invalid"
                        )
                        continue
                    page_number = entry.get("page_number")
                    reason = entry.get("reason")
                    if (
                        not isinstance(page_number, int)
                        or isinstance(page_number, bool)
                        or page_number in pages
                        or not isinstance(reason, str)
                        or not reason.strip()
                    ):
                        errs.append(
                            f"INV-11 {doc_id}: non-evidence page annotation is invalid"
                        )
                        continue
                    pages.add(page_number)
                excluded_pages_by_document[doc_id] = pages

        source_doc_ids = {
            source.get("doc_id")
            for source in sources
            if isinstance(source.get("doc_id"), str)
        }
        if eligibility is not None and (
            set(excluded_pages_by_document) - source_doc_ids
        ):
            errs.append("INV-11 non-evidence pages reference an unknown source")
        for source in sources:
            doc_id = source.get("doc_id")
            page_count = source.get("page_count")
            if (
                not isinstance(page_count, int)
                or source.get("indexed_page_numbers")
                != list(range(1, page_count + 1))
            ):
                errs.append(
                    f"INV-11 {doc_id}: not every source page was indexed"
                )
            if eligibility is not None and isinstance(page_count, int):
                excluded_pages = sorted(
                    excluded_pages_by_document.get(doc_id, set())
                )
                eligible_pages = [
                    page_number
                    for page_number in range(1, page_count + 1)
                    if page_number not in excluded_pages
                ]
                source_eligible = source.get(
                    "eligible_candidate_chunk_count"
                )
                source_excluded = source.get(
                    "excluded_candidate_chunk_count"
                )
                if (
                    source.get("non_evidence_page_numbers")
                    != excluded_pages
                    or source.get("evidence_eligible_page_numbers")
                    != eligible_pages
                    or not isinstance(source_eligible, int)
                    or not isinstance(source_excluded, int)
                    or source_eligible + source_excluded
                    != source.get("candidate_chunk_count")
                ):
                    errs.append(
                        f"INV-11 {doc_id}: evidence eligibility summary is inconsistent"
                    )
        if eligibility is not None and (
            sum(
                source.get("eligible_candidate_chunk_count", 0)
                for source in sources
            )
            != eligible_count
            or sum(
                source.get("excluded_candidate_chunk_count", 0)
                for source in sources
            )
            != excluded_count
        ):
            errs.append("INV-11 source evidence-eligibility counts are inconsistent")
        for chunk in selected_chunks:
            if chunk.get("page_number") in excluded_pages_by_document.get(
                chunk.get("doc_id"),
                set(),
            ):
                errs.append("INV-11 a selected chunk comes from a non-evidence page")
        if retrieval.get("source_pdf_verification_required") is True and (
            retrieval.get("source_pdf_verification", {}).get("status")
            != "verified"
        ):
            errs.append("INV-11 required source PDF verification is missing")

        documents = t.get("document_set", {}).get("documents", [])
        document_chunk_ids = []
        document_tokens = 0
        for document in documents:
            metadata = document.get("retrieval")
            if not isinstance(metadata, dict):
                errs.append(
                    f"INV-11 {document.get('doc_id')}: retrieval metadata is missing"
                )
                continue
            if metadata.get("profile_id") != retrieval.get("profile_id"):
                errs.append(
                    f"INV-11 {document.get('doc_id')}: retrieval profile differs"
                )
            document_chunk_ids.extend(metadata.get("selected_chunk_ids", []))
            document_tokens += metadata.get("selected_token_count", 0)
        if document_chunk_ids != selected_ids:
            errs.append("INV-11 document chunk IDs do not match the retrieval trace")
        if document_tokens != selected_tokens:
            errs.append("INV-11 document token counts do not match the retrieval trace")
        document_budgets = retrieval.get("document_token_budgets")
        if document_budgets is not None:
            document_ids = {
                document.get("doc_id")
                for document in documents
                if isinstance(document.get("doc_id"), str)
            }
            if (
                not isinstance(document_budgets, dict)
                or set(document_budgets) != document_ids
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 1
                    for value in document_budgets.values()
                )
                or sum(document_budgets.values())
                > retrieval.get("document_token_budget", -1)
            ):
                errs.append("INV-11 per-document token budgets are invalid")
            else:
                selected_tokens_by_document = {
                    doc_id: sum(
                        chunk.get("token_count", 0)
                        for chunk in selected_chunks
                        if chunk.get("doc_id") == doc_id
                    )
                    for doc_id in document_ids
                }
                if any(
                    selected_tokens_by_document[doc_id] > budget
                    for doc_id, budget in document_budgets.items()
                ):
                    errs.append(
                        "INV-11 selected chunks exceed a per-document token budget"
                    )

        carrier = next(
            (document for document in documents
             if document.get("role") == "injection_carrier"),
            None,
        )
        if carrier is None or retrieval.get("injection_bearing_chunk_id") not in (
            carrier.get("retrieval", {}).get("selected_chunk_ids", [])
        ):
            errs.append("INV-11 injection-bearing chunk is not in the carrier view")
        injected_text = t.get("injection", {}).get("injected_text")
        payload_occurrences = (
            sum(document.get("text", "").count(injected_text) for document in documents)
            if isinstance(injected_text, str) and injected_text
            else 0
        )
        expected_occurrences = 1 if t.get("treatment") == "injected" else 0
        if payload_occurrences != expected_occurrences:
            errs.append(
                "INV-11 retrieved views contain the wrong number of injection payloads"
            )

        retrieval_events = [
            event for event in t.get("trajectory_trace", {}).get("full_events", [])
            if event.get("type") == "tool_call"
            and event.get("tool_name") == "retrieve_documents"
        ]
        if len(retrieval_events) != 1:
            errs.append("INV-11 trajectory must contain one retrieval event")
        else:
            metrics = retrieval_events[0].get("retrieval_metrics", {})
            if metrics.get("selected_chunk_ids") != selected_ids:
                errs.append("INV-11 retrieval event selected IDs are inconsistent")
            if metrics.get("ranking_used_injection_text") is not False:
                errs.append("INV-11 retrieval event is not treatment-blind")
            metric_fields = (
                "retrieval_profile_id",
                "canonical_ranking_treatment",
                "candidate_chunk_count",
                "eligible_candidate_chunk_count",
                "excluded_candidate_chunk_count",
                "selected_chunk_count",
                "selected_token_count",
                "document_token_budget",
                "document_token_budgets",
            )
            retrieval_fields = {
                "retrieval_profile_id": retrieval.get("profile_id"),
                "canonical_ranking_treatment": retrieval.get(
                    "canonical_ranking_treatment"
                ),
                "candidate_chunk_count": candidate_count,
                "eligible_candidate_chunk_count": eligible_count,
                "excluded_candidate_chunk_count": excluded_count,
                "selected_chunk_count": retrieval.get("selected_chunk_count"),
                "selected_token_count": retrieval.get("selected_token_count"),
                "document_token_budget": retrieval.get(
                    "document_token_budget"
                ),
                "document_token_budgets": document_budgets,
            }
            if any(
                metrics.get(field) != retrieval_fields[field]
                for field in metric_fields
            ):
                errs.append("INV-11 retrieval event metrics are inconsistent")
    return errs


def validate_payload(t):
    """Return structural and semantic errors for an in-memory trajectory."""

    with open(SCHEMA_PATH) as handle:
        schema = json.load(handle)
    validator = Draft202012Validator(schema)
    struct = sorted(validator.iter_errors(t), key=lambda error: error.path)
    struct_msgs = [
        f"SCHEMA {'/'.join(map(str, error.path))}: {error.message}"
        for error in struct
    ]
    return struct_msgs + (semantic_checks(t) if not struct_msgs else [])


def validate_file(path, validator):
    with open(path) as f:
        t = json.load(f)
    struct = sorted(validator.iter_errors(t), key=lambda e: e.path)
    struct_msgs = [f"SCHEMA {'/'.join(map(str, e.path))}: {e.message}" for e in struct]
    sem_msgs = semantic_checks(t) if not struct_msgs else []
    ok = not struct_msgs and not sem_msgs
    return ok, struct_msgs + sem_msgs


def main():
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    validator = Draft202012Validator(schema)
    paths = []
    default_glob = os.path.join(
        PROJECT_ROOT, "experiments", "scenario1", "trajectories", "*.json"
    )
    for arg in (sys.argv[1:] or [default_glob]):
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        print("no files matched")
        sys.exit(2)
    all_ok = True
    for p in paths:
        ok, msgs = validate_file(p, validator)
        print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(p)}")
        for m in msgs:
            print(f"      - {m}")
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
