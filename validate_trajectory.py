#!/usr/bin/env python3
"""
Validate scenario 1 trajectory files.

Two layers:
  1. Structural: JSON Schema (scenario1_trajectory.schema.json).
  2. Semantic invariants that JSON Schema cannot express and that map to the
     bugs this project actually hit:
       INV-1  the primary layer is present in every extraction (the 1B-model
              bug dropped layer 20).
       INV-2  exactly the agents in raw_poison_exposed_agents have the injection
              in their prompt; everyone else must not (token-alignment bug +
              executor re-reading raw poison).
       INV-3  hops_survived == unsafe_action_hop_index - injection_hop_index.
       INV-4  delegation_depth == condition_id, and hop_path length matches.
       INV-5  action_channel.unsafe_action_executed and the deprecated aliases
              agree with the unsafe_action events (skipped while the value is
              null, i.e. before the model has run).
       INV-6  no auto-derived step_label is ever reasoning_compromised (that
              state is annotation-only).
       INV-7  outcome_class is coherent with treatment and the channels:
              injection_present false forces clean; executed iff the action
              fired; propagated_but_not_executed is output_adoption true with
              the action not fired.
       INV-8  agent topology matches the depth condition (a 3-hop trace must
              contain worker_2; no unexpected agents).
       INV-9  a non-null reasoning label requires human or mechanistic evidence;
              construction/auto proxies may not set it.

Usage: python validate_trajectory.py experiments/scenario1/trajectories/*.json
"""

import glob
import json
import sys
import os

from jsonschema import Draft202012Validator

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "scenario1_trajectory.schema.json")
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

    # INV-1: primary layer present in every extraction block.
    primary = t["activation_config"]["primary_layer"]
    for e in _agent_turns(t):
        for key in ("activation_metadata", "attention_metadata"):
            meta = e.get(key)
            if meta and primary not in meta.get("layers_extracted", []):
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

    # INV-3: hops_survived arithmetic.
    cp = t["compromise_propagation"]
    if cp.get("injection_hop_index") is not None and cp.get("hops_survived") is not None:
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
        if not injected and oc != "clean":
            errs.append(f"INV-7 outcome_class {oc} but injection_present is false (must be clean)")
        if injected and oc == "clean":
            errs.append("INV-7 outcome_class clean but injection_present is true")
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
    return errs


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
    for arg in (sys.argv[1:] or [os.path.join(HERE, "experiments/scenario1/trajectories/*.json")]):
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
