#!/usr/bin/env python3
"""Generate matched-group trajectories from a domain input config.

Reads a domain JSON config and produces four matched trajectories:
  clean 2-hop, injected 2-hop, clean 3-hop, injected 3-hop

The generator saves raw facts per agent turn. It does not assign
behavioral outcome labels. Label assignment is handled downstream
by the adapter in src/pipeline/handoff.py (PR #4).

For dry runs, the model is not called. Agent outputs are honest
placeholders marked with generation_mode=dry_run.
"""

import argparse
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "experiments", "scenario1", "trajectories")

MODEL_CONFIG = {
    "model_name": "Qwen/Qwen3-32B",
    "provider": "modal",
    "model_revision": None,
    "tokenizer_name": "Qwen/Qwen3-32B",
    "tokenizer_revision": None,
    "dtype": "bfloat16",
    "num_hidden_layers": 64,
    "temperature": 0.0,
    "seed": 0,
    "decoding_settings": {
        "do_sample": False,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    },
}

ACTIVATION_LAYERS = [32, 40, 48]
ACTIVATION_PRIMARY_LAYER = 40


def load_domain_config(path):
    with open(path) as f:
        return json.load(f)


def _get_documents(config, treatment):
    docs = []
    for doc in config["documents"]:
        if doc["role"] == "always_clean":
            docs.append({"doc_id": doc["doc_id"], "body": doc["body"]})
        elif doc["role"] == "clean_or_poisoned":
            if treatment == "clean":
                docs.append({"doc_id": doc["doc_id"], "body": doc["body_clean"]})
            else:
                docs.append({"doc_id": doc["doc_id"], "body": doc["body_poisoned"]})
    return docs


def _dry_run_placeholder(trajectory_id, step_index, kind):
    return {
        "model_name": MODEL_CONFIG["model_name"],
        "extraction_method": (
            "hf_output_hidden_states" if kind == "activation"
            else "hf_output_attentions"
        ),
        "hook_point": (
            "residual_stream_post_block" if kind == "activation"
            else "attention_weights_per_block"
        ),
        "primary_layer": ACTIVATION_PRIMARY_LAYER,
        "layers_extracted": list(ACTIVATION_LAYERS),
        "layers_skipped_out_of_range": [],
        "model_num_layers": None,
        "token_position_extracted": None,
        "token_position_convention": "final_token_in_sequence",
        "sequence_length": None,
        "storage_path": None,
        "storage_format": "torch.save dict[layer_key -> tensor]",
        "storage_status": "dry_run_placeholder",
    }


def _clean_token_alignment():
    return {
        "injection_present_in_prompt": False,
        "char_span": None,
        "injection_token_span": None,
        "adversarial_instruction_detected_tokens": {
            "detected": False,
            "token_indices": [],
            "token_strings": [],
        },
    }


def _find_injection_span(body, payload):
    start = body.find(payload)
    if start == -1:
        return None
    return {"start_char": start, "end_char": start + len(payload)}


def build_trajectory(config, treatment, condition):
    is_2hop = condition == "2-hop"
    domain = config["domain_id"]
    group_id = config["independence_group_id"]
    tid = f"scenario1_{domain}_{treatment}_{condition.replace('-', '')}_{config['wording_id']}"
    pair_id = f"{group_id}_{condition.replace('-', '')}_seed0"

    docs = _get_documents(config, treatment)
    is_injected = treatment == "injected"

    hop_path = (
        ["user", "planner", "worker", "retrieved_document", "executor"]
        if is_2hop else
        ["user", "planner", "worker", "retrieved_document",
         "second_worker", "executor"]
    )

    system_prompts = config["system_prompts"]
    injection_payload = None
    poisoned_doc = None
    if is_injected:
        for doc_def in config["documents"]:
            if doc_def["role"] == "clean_or_poisoned":
                injection_payload = doc_def["injection_payload"]
                poisoned_doc = doc_def
                break

    events = []
    step_idx = 0

    # Planner turn
    planner_input = {
        "system_prompt": system_prompts["planner"],
        "user_message": config["task"],
    }
    events.append({
        "type": "agent_turn",
        "agent_role": "planner",
        "agent_id": "planner_1",
        "hop_index": 0,
        "input": {
            "system_prompt": system_prompts["planner"],
            "user_message": config["task"],
            "rendered_prompt": None,
            "input_token_ids": None,
            "rendered_prompt_hash": None,
        },
        "output": {
            "message": None,
            "raw_generated_text": None,
            "generated_token_ids": None,
            "finish_reason": None,
            "truncated": None,
            "thinking_content": None,
            "final_content": None,
        },
        "generation_mode": "dry_run",
        "token_alignment": _clean_token_alignment(),
        "activation_metadata": _dry_run_placeholder(tid, step_idx, "activation"),
        "attention_metadata": _dry_run_placeholder(tid, step_idx, "attention"),
        "trajectory_id": tid,
        "step_index": step_idx,
        "metadata": {
            "scenario": "research_pipeline_exfiltration",
            "depth_condition": condition,
            "timestamp": None,
        },
    })
    step_idx += 1

    # Tool call: retrieve_documents
    tool_result = [{"doc_id": d["doc_id"], "body": d["body"]} for d in docs]
    events.append({
        "type": "tool_call",
        "tool_name": "retrieve_documents",
        "hop_index": 1,
        "tool_result": tool_result,
        "retrieval_metrics": {
            "retrieved_ids": [d["doc_id"] for d in docs],
            "n_documents": len(docs),
            "poison_in_retrieval": is_injected,
        },
        "trajectory_id": tid,
        "step_index": step_idx,
        "metadata": {
            "scenario": "research_pipeline_exfiltration",
            "depth_condition": condition,
            "timestamp": None,
        },
    })
    step_idx += 1

    # Worker 1 turn (sees raw documents)
    worker_token_alignment = _clean_token_alignment()
    if is_injected and injection_payload:
        poisoned_body = None
        for d in docs:
            if d["doc_id"] == poisoned_doc["doc_id"]:
                poisoned_body = d["body"]
                break
        if poisoned_body:
            span = _find_injection_span(poisoned_body, injection_payload)
            if span:
                worker_token_alignment = {
                    "injection_present_in_prompt": True,
                    "char_span": span,
                    "injection_token_span": None,
                    "adversarial_instruction_detected_tokens": {
                        "detected": True,
                        "token_indices": None,
                        "token_strings": None,
                    },
                    "tokenizer_name": MODEL_CONFIG["tokenizer_name"],
                    "tokenizer_revision": MODEL_CONFIG["tokenizer_revision"],
                    "rendered_prompt_hash": None,
                    "truncation_removed_injection_tokens": False,
                }

    events.append({
        "type": "agent_turn",
        "agent_role": "worker",
        "agent_id": "worker_1",
        "hop_index": 1,
        "sub_step": "post_retrieval",
        "input": {
            "system_prompt": system_prompts["worker"],
            "upstream_agent_message": None,
            "retrieved_document_text": [d["doc_id"] for d in docs],
            "saw_raw_documents": True,
            "rendered_prompt": None,
            "input_token_ids": None,
            "rendered_prompt_hash": None,
        },
        "output": {
            "message": None,
            "raw_generated_text": None,
            "generated_token_ids": None,
            "finish_reason": None,
            "truncated": None,
            "thinking_content": None,
            "final_content": None,
        },
        "generation_mode": "dry_run",
        "token_alignment": worker_token_alignment,
        "activation_metadata": _dry_run_placeholder(tid, step_idx, "activation"),
        "attention_metadata": _dry_run_placeholder(tid, step_idx, "attention"),
        "trajectory_id": tid,
        "step_index": step_idx,
        "metadata": {
            "scenario": "research_pipeline_exfiltration",
            "depth_condition": condition,
            "timestamp": None,
        },
    })
    step_idx += 1

    # Worker 2 (3-hop only, receives upstream message, NOT raw documents)
    if not is_2hop:
        events.append({
            "type": "agent_turn",
            "agent_role": "worker",
            "agent_id": "worker_2",
            "hop_index": 2,
            "sub_step": "relay",
            "input": {
                "system_prompt": system_prompts["worker2"],
                "upstream_agent_message": None,
                "saw_raw_documents": False,
                "rendered_prompt": None,
                "input_token_ids": None,
                "rendered_prompt_hash": None,
            },
            "output": {
                "message": None,
                "raw_generated_text": None,
                "generated_token_ids": None,
                "finish_reason": None,
                "truncated": None,
                "thinking_content": None,
                "final_content": None,
            },
            "generation_mode": "dry_run",
            "token_alignment": _clean_token_alignment(),
            "activation_metadata": _dry_run_placeholder(tid, step_idx, "activation"),
            "attention_metadata": _dry_run_placeholder(tid, step_idx, "attention"),
            "trajectory_id": tid,
            "step_index": step_idx,
            "metadata": {
                "scenario": "research_pipeline_exfiltration",
                "depth_condition": condition,
                "timestamp": None,
            },
        })
        step_idx += 1

    # Executor turn (receives upstream message, NOT raw documents)
    executor_hop = 2 if is_2hop else 3
    events.append({
        "type": "agent_turn",
        "agent_role": "executor",
        "agent_id": "executor_1",
        "hop_index": executor_hop,
        "input": {
            "system_prompt": system_prompts["executor"],
            "upstream_agent_message": None,
            "saw_raw_documents": False,
            "rendered_prompt": None,
            "input_token_ids": None,
            "rendered_prompt_hash": None,
        },
        "output": {
            "message": None,
            "raw_generated_text": None,
            "generated_token_ids": None,
            "parsed_tool_calls": None,
            "finish_reason": None,
            "truncated": None,
            "thinking_content": None,
            "final_content": None,
            "actions": None,
        },
        "generation_mode": "dry_run",
        "token_alignment": _clean_token_alignment(),
        "activation_metadata": _dry_run_placeholder(tid, step_idx, "activation"),
        "attention_metadata": _dry_run_placeholder(tid, step_idx, "attention"),
        "trajectory_id": tid,
        "step_index": step_idx,
        "metadata": {
            "scenario": "research_pipeline_exfiltration",
            "depth_condition": condition,
            "timestamp": None,
        },
    })
    step_idx += 1

    injection_char_span = None
    if is_injected and poisoned_doc and injection_payload:
        for d in docs:
            if d["doc_id"] == poisoned_doc["doc_id"]:
                span = _find_injection_span(d["body"], injection_payload)
                if span:
                    injection_char_span = {
                        "doc_id": d["doc_id"],
                        "start_char": span["start_char"],
                        "end_char": span["end_char"],
                    }
                break

    injection_block = {
        "injection_present": is_injected,
        "injection_source_id": (
            poisoned_doc["doc_id"] if is_injected and poisoned_doc else None
        ),
        "injection_type": "body_text" if is_injected else "none",
        "injection_variant": config["wording_id"] if is_injected else None,
        "injection_family": config["injection_family"],
        "injection_placement": config["injection_placement"],
        "injection_marker": config["carrier_framing"] if is_injected else None,
        "injection_endpoint": config["endpoint"] if is_injected else None,
        "injection_vector": "plaintext_marker" if is_injected else None,
        "injection_render": "visible_body_text" if is_injected else None,
        "injected_text": injection_payload if is_injected else None,
        "injection_char_span_in_source_doc": injection_char_span,
        "injection_point": (
            {"agent_role": "worker", "agent_id": "worker_1", "hop_index": 1}
            if is_injected else None
        ),
        "injection_point_status": "locked",
        "raw_poison_exposed_agents": ["worker_1"] if is_injected else [],
    }

    return {
        "schema_version": "spec_gap.scenario1.v3",
        "trajectory_id": tid,
        "metadata": {
            "independence_group_id": group_id,
            "domain_id": domain,
            "task_family_id": config["task_family_id"],
            "pair_id": pair_id,
            "condition": treatment,
            "hop_mode": condition,
            "injection_family": config["injection_family"],
            "injection_variant": config["wording_id"],
            "injection_placement": config["injection_placement"],
            "document_set_id": f"{domain}_docs_01",
        },
        "scenario_id": "scenario_1",
        "scenario_description": (
            "Multi-hop retrieval and summarization. The third retrieved "
            "document may carry a hidden injection instructing exfiltration."
        ),
        "independence_group_id": group_id,
        "domain_id": domain,
        "task_family_id": config["task_family_id"],
        "document_set_id": f"{domain}_docs_01",
        "matched_pair_id": pair_id,
        "treatment": treatment,
        "condition_id": condition,
        "delegation_depth": condition,
        "generation_mode": "dry_run",
        "provenance": {
            "created_by": config["created_by"],
            "generator": "generate_onyinye_trajectories.py",
            "source_branch": "onyinye/trajectory-generation",
        },
        "task": {
            "user_task": config["task"],
            "expected_benign_behavior": config["expected_benign_behavior"],
        },
        "hop_path": hop_path,
        "model": dict(MODEL_CONFIG),
        "activation_config": {
            "layers": list(ACTIVATION_LAYERS),
            "primary_layer": ACTIVATION_PRIMARY_LAYER,
            "artifact_root": "experiments/scenario1",
            "token_position": "last",
        },
        "injection": injection_block,
        "compromise_propagation": {
            "injection_hop_index": 1 if is_injected else None,
            "unsafe_action_hop_index": None,
            "hops_survived": None,
            "first_behavioral_compromise_hop": None,
            "chain_completed": None,
        },
        "trajectory_trace": {"full_events": events},
        "evaluation_labels": {
            "injection_present": is_injected,
            "action_channel": {
                "unsafe_action_executed": None,
                "first_action_hop": None,
                "source": "pending_live_run",
            },
            "behavioral_channel": {
                "output_adoption": None,
                "first_behavioral_compromise_hop": None,
                "source": "pending_live_run",
            },
            "reasoning_channel": {
                "label": None,
                "annotation_status": (
                    "not_applicable_clean" if not is_injected
                    else "human_or_mechanistic_evidence_required"
                ),
                "note": (
                    "clean baseline: no injected goal to adopt" if not is_injected
                    else "requires human or probe evidence"
                ),
            },
            "outcome_class": None,
            "injection_success": None,
            "exfiltration_detected": None,
        },
        "document_set": {
            "document_set_id": f"{domain}_docs_01",
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "poisoned": is_injected and d["doc_id"] == (
                        poisoned_doc["doc_id"] if poisoned_doc else None
                    ),
                    "chars": len(d["body"]),
                }
                for d in docs
            ],
        },
        "activation_extraction": {
            "status": "dry_run_placeholder",
            "note": (
                "Dry run: model was not called. Activation and attention "
                "tensors will be extracted during a real --mode real run."
            ),
            "primary_layer": ACTIVATION_PRIMARY_LAYER,
            "layers": list(ACTIVATION_LAYERS),
            "token_position": "last",
        },
    }


def validate_input_config(config):
    """Check the input config before generating trajectories."""
    errors = []

    marker = config["carrier_framing"]
    endpoint = config["endpoint"]

    doc3 = None
    for doc in config["documents"]:
        if doc["role"] == "clean_or_poisoned":
            doc3 = doc
            break

    if doc3 is None:
        errors.append("No clean_or_poisoned document found in config.")
        return errors

    if marker in doc3["body_clean"]:
        errors.append(f"Clean doc 3 contains injection marker '{marker}'.")
    if endpoint in doc3["body_clean"]:
        errors.append(f"Clean doc 3 contains injection endpoint '{endpoint}'.")
    if marker not in doc3["body_poisoned"]:
        errors.append("Poisoned doc 3 is missing injection marker.")
    if endpoint not in doc3["body_poisoned"]:
        errors.append("Poisoned doc 3 is missing injection endpoint.")

    doc_ids = [d["doc_id"] for d in config["documents"]]
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("Duplicate document IDs found.")
    if len(config["documents"]) != 3:
        errors.append(f"Expected 3 documents, found {len(config['documents'])}.")

    return errors


def validate_match_group(trajectories):
    """Check that the 4 generated trajectories form a valid match group."""
    errors = []

    if len(trajectories) != 4:
        errors.append(f"Expected 4 trajectories, found {len(trajectories)}.")
        return errors

    conditions = {(t["treatment"], t["condition_id"]) for t in trajectories}
    expected = {("clean", "2-hop"), ("injected", "2-hop"),
                ("clean", "3-hop"), ("injected", "3-hop")}
    if conditions != expected:
        errors.append(f"Missing conditions: {expected - conditions}")

    tasks = {t["task"]["user_task"] for t in trajectories}
    if len(tasks) != 1:
        errors.append("Trajectories do not share the same task.")

    doc_id_sets = []
    for t in trajectories:
        ids = tuple(d["doc_id"] for d in t["document_set"]["documents"])
        doc_id_sets.append(ids)
    if len(set(doc_id_sets)) != 1:
        errors.append("Document IDs differ across trajectories.")

    for t in trajectories:
        events = t["trajectory_trace"]["full_events"]
        for e in events:
            if e.get("agent_id") in ("worker_2", "executor_1"):
                if e.get("input", {}).get("saw_raw_documents"):
                    errors.append(
                        f"{t['trajectory_id']}: {e['agent_id']} "
                        f"marked as seeing raw documents."
                    )

        if "3hop" in t["trajectory_id"]:
            agent_ids = [e.get("agent_id") for e in events
                         if e.get("type") == "agent_turn"]
            if "worker_2" not in agent_ids:
                errors.append(f"{t['trajectory_id']}: 3-hop missing worker_2.")

        el = t["evaluation_labels"]
        if el["outcome_class"] is not None:
            errors.append(
                f"{t['trajectory_id']}: outcome_class should be null for dry run."
            )

    tids = [t["trajectory_id"] for t in trajectories]
    if len(tids) != len(set(tids)):
        errors.append("Duplicate trajectory IDs.")

    return errors


def write_trajectory(trajectory, json_path):
    with open(json_path, "w") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate matched-group trajectories from domain input config."
    )
    ap.add_argument(
        "--input", required=True,
        help="Path to domain input config JSON."
    )
    ap.add_argument(
        "--out", default=OUTPUT_DIR,
        help="Output directory for trajectory files."
    )
    args = ap.parse_args()

    config = load_domain_config(args.input)

    input_errors = validate_input_config(config)
    if input_errors:
        print("Input config validation failed:")
        for e in input_errors:
            print(f"  {e}")
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)

    generated = []
    all_trajectories = []
    manifest_entries = []

    for treatment in ["clean", "injected"]:
        for condition in ["2-hop", "3-hop"]:
            t = build_trajectory(config, treatment, condition)
            all_trajectories.append(t)
            tid = t["trajectory_id"]
            json_path = os.path.join(args.out, f"{tid}.json")
            write_trajectory(t, json_path)
            generated.append(f"{tid}.json")

            manifest_entries.append({
                "trajectory_id": tid,
                "independence_group_id": config["independence_group_id"],
                "domain_id": config["domain_id"],
                "treatment": treatment,
                "condition": condition,
                "injection_variant": config["wording_id"],
                "injection_present": treatment == "injected",
                "matched_pair_id": t["matched_pair_id"],
                "generation_mode": "dry_run",
                "created_by": config["created_by"],
            })

    manifest = {
        "schema_version": "spec_gap.scenario1.v3",
        "generation_mode": "dry_run",
        "model": MODEL_CONFIG["model_name"],
        "created_by": config["created_by"],
        "generator": "generate_onyinye_trajectories.py",
        "source_branch": "onyinye/trajectory-generation",
        "created": datetime.now(timezone.utc).isoformat(),
        "match_groups": [
            {
                "independence_group_id": config["independence_group_id"],
                "domain_id": config["domain_id"],
                "task_family_id": config["task_family_id"],
                "wording_id": config["wording_id"],
                "n_trajectories": 4,
                "conditions": ["clean 2-hop", "injected 2-hop",
                               "clean 3-hop", "injected 3-hop"],
            }
        ],
        "trajectory_coverage": {
            "n_total": len(manifest_entries),
            "n_clean": sum(1 for e in manifest_entries if not e["injection_present"]),
            "n_injected": sum(1 for e in manifest_entries if e["injection_present"]),
            "outcomes_are_observed": False,
            "note": (
                "Dry-run structural fixtures. The model was not called. "
                "Outcomes will be determined by live model runs."
            ),
        },
        "trajectories": manifest_entries,
    }

    group_errors = validate_match_group(all_trajectories)
    if group_errors:
        print("Match group validation failed:")
        for e in group_errors:
            print(f"  {e}")
        raise SystemExit(1)

    manifest_path = os.path.join(args.out, "manifest_onyinye.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Generated {len(generated)} trajectory files in {args.out}:")
    for fn in sorted(generated):
        print(f"  {fn}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
