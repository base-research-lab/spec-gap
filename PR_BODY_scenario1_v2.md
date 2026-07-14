## Summary

Replacement for PR #7, opened fresh from `main` as a **draft**. This is the
Scenario 1 **schema + controlled-inputs scaffold** for the public-health match
group, in the shared event-style format. It is not the full dataset and not a
model runner. Reference: PR #7 (to be closed once this is open).

Schema version: **`spec_gap.scenario1.v2`** (matches PR #4's normalization path).

## Files owned by this PR

- `scenario1_trajectory.schema.json` — one v2 schema; six canonical outcomes.
- `validate_trajectory.py` — the single (lowercase) validator: JSON Schema + 9
  semantic invariants.
- `experiments/scenario1/inputs/` — `registry.json`, three clean document
  fixtures named after the source PDFs (Report 3 is the clean canonical text;
  the injection is inserted only for the injected condition), and the source
  PDFs.
- `scenario1_pipeline.py` — structural **dry-run generator only** (no model, no
  GPU, no `.pt`).
- `tests/test_scenario1_schema.py`, `tests/test_scenario1_validator.py`.
- `pyproject.toml` — adds `jsonschema`.

Deliberately **not** included: any real-model backend, `live_run_config.py`,
the old pilot JSONL, the duplicate capital validator, or PDFs used as poisoned
"clean" documents.

## Ownership / interfaces expected

- Generation: Onyinye's shared generator (reads the input registry).
- Model execution: **PR #8** (`Qwen/Qwen3-32B` on Modal, thinking on/off,
  activation extraction, model metadata). The schema holds exact model I/O and
  honest activation references; extraction layers stay configurable, no primary
  layer is locked.
- Normalization: **PR #4** adapter (six canonical outcomes, v2 contract).

## The four base conditions (one match group)

`ph_breast_cancer_A`: clean 2-hop, injected 2-hop, clean 3-hop, injected 3-hop.
All four share the same task, documents, system prompts, wording (A), seed, and
document order — only treatment and depth vary. Injection is fixed at `worker_1`
at both depths; `worker_2` and the executor are relay-fed. The injected executor
forwards the endpoint in text but fires no structured tool call, so the outcome
is `propagated_but_not_executed` (not `executed`). Reasoning labels stay null.

## Commands run and results

```
python scenario1_pipeline.py --mode dry_run
python validate_trajectory.py experiments/scenario1/trajectories/*.json
# -> PASS x4 (clean/injected x 2-hop/3-hop)

python -m pytest tests/test_scenario1_schema.py tests/test_scenario1_validator.py -q
# -> 22 passed
```

The dry-run trajectory files are an **initial end-to-end validation subset**:
they confirm the pipeline and schema agree. A dry run does not call the model,
does not start a GPU, and does not create activation files. **They are not
real-model experiment results.** Generated trajectories and the manifest are
gitignored.

## Open integration items

- Confirm the six outcome names and the v2 field contract against PR #4.
- Confirm the `registry.json` shape against Onyinye's shared input registry.
- Confirm the real-model I/O field names with PR #8.
- Swap in the team's real cleaned Report 3 text for the placeholder fixture.
