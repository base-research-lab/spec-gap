# Running Qwen3-32B on Modal

This guide covers runbook stages `S07`–`S13`. The Modal backend owns one model
turn at a time: generation, exact input/output capture, residual-stream
extraction, and cost logging. It does not construct the Scenario 1 dataset or
decide whether a simulated action executed. Start with the
[canonical pipeline runbook](scenario1/pipeline-runbook.md); the older numbers
inside script filenames are stable phase-local names, not a second run order.

## Workspace and resources

The reference backend runs in the workspace selected by the active Modal
profile. The app creates or reuses:

- app: `spec-gap-qwen3-32b`;
- model volume: `spec-gap-qwen3-32b-model`;
- activation volume: `spec-gap-scenario1-artifacts`.

These names are workspace-local. The code does not depend on one person's
filesystem or require the original lab workspace: an authorized contributor
may select another workspace, where Modal creates separate named Volumes. A
new workspace starts with an empty model cache and has its own access, storage,
and compute billing.

The active model contract is:

```text
model: Qwen/Qwen3-32B
revision: 9216db5781bf21249d130ec9da846c4624c16137
GPU: 1 x H200 for a paid model call
maximum active containers: 1
```

The revision is pinned in code and was resolved when the shared model cache was
created. Do not silently replace it with a newer `main` revision during the
thinking-mode comparison.

## 1. Confirm the workspace

Modal credentials are user- and workspace-specific and must never be committed
to this repository. From the project environment, authenticate once and inspect
the selected workspace:

```bash
modal setup
modal token info
modal profile current
```

If the executable is not on `PATH`, run `python -m modal setup` or activate the
virtual environment created during installation. CI may instead provide
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` through its secret store.

Confirm that the selected profile points to the workspace where the model and
artifact volumes should live.

Modal functions are serverless. A completed run does not leave a RunPod-style
GPU pod for the user to terminate. The runner allows the warm container to
scale down for reuse, then Modal releases it automatically. To inspect recent
state, run `modal app list`; a stopped app or an app with zero active tasks is
not using an H200.

## 2. Validate locally and check credentials

The repository-wide portable check builds both supported construction cohorts
in temporary storage and optionally verifies the selected Modal credentials:

```bash
python scripts/run_portable_smoke_test.py --check-modal
```

It covers 9 domains (81 grid registries), 324 structural trajectories, and 2268 request templates.
The access check runs only `modal token info` and read-only `modal app list
--json`; it does not register the production app, build its image, call Qwen,
or allocate a GPU.

The following lower-level command enters Modal's app lifecycle. `--action
validate` never calls Qwen or allocates a GPU, but Modal may prepare the app
image and named resources, so workspace usage may occur:

```bash
modal run scripts/02_model_execution/03_modal_qwen_runner.py \
  --request-path tests/fixtures/qwen_agent_turn_request.json \
  --action validate
```

The fixture is infrastructure-only and must not be added to the Scenario 1
manifest.

## 3. Cache the model without a GPU

```bash
modal run scripts/02_model_execution/03_modal_qwen_runner.py \
  --request-path tests/fixtures/qwen_agent_turn_request.json \
  --action download \
  --model-revision 9216db5781bf21249d130ec9da846c4624c16137
```

This uses remote CPU, memory, network, and Volume storage. It does not start an
H200. Run it once when the selected workspace's model Volume is new or has been
removed. An existing workspace may already contain the pinned revision, but a
contributor must not assume that a cache from another workspace is available.

## 4. Run one paid model turn

Use a real request produced from a Scenario 1 match group, not the test fixture:

```bash
modal run scripts/02_model_execution/03_modal_qwen_runner.py \
  --request-path path/to/real_agent_turn_request.json \
  --action run \
  --analysis-tier exploratory \
  --confirm-paid-run RUN_H200 \
  --output-path path/to/model_turn_output.json
```

Both `--action run` and `--confirm-paid-run RUN_H200` are required. A result is
not a complete trajectory: the next agent request must be built from the saved
downstream message, and the safe simulated executor must separately record the
action result.

## Thinking comparison

The controlled comparison changes only `enable_thinking`:

```text
do_sample=true
temperature=0.6
top_p=0.95
top_k=20
min_p=0.0
seed=0
```

Thinking content is saved for analysis but is not forwarded downstream. Only
the visible final response and valid requested tool calls may reach the next
agent.

## Generation-budget versions

Generation budgets are explicit protocol versions rather than silent runtime
overrides:

- `controlled_v1_2048` preserves the original 2,048-token pilot runs and their
  historical artifact IDs.
- `controlled_v2_5000` changes only `max_new_tokens` to 5,000. Its trajectory,
  checkpoint, activation, and cost IDs include
  `__gen_controlled_v2_5000`.

The registry selects the protocol, and its retrieval plan must reserve the same
output budget. The exact Qwen chat-template preflight must pass before a paid
run. Definitive v2 analysis reruns every selected clean/injected condition;
only rerunning previously truncated outputs would create selective-rerun bias.
Any later turn that reaches 5,000 remains recorded as truncated and
indeterminate rather than receiving an individual budget increase.

Seeds are protocol-locked: the registry seed must equal the versioned
generation protocol seed. Testing a different seed requires a new protocol ID
and a complete matched rerun, rather than an in-place registry edit. The July
31 staged v2 matrix combined an exploratory smoke with a later definitive run,
so it does not satisfy the uniformly definitive rule above and is not yet the
paper-analysis base.

## Sampling and replay provenance

The registry seed is copied into the structural record, decoding settings,
generation request, saved result, and live agent-turn metadata. Immediately
before each sampled turn, the runner calls both `torch.manual_seed` and
`torch.cuda.manual_seed_all`. New cost records also capture the Torch,
Transformers, CUDA, GPU, and deterministic-algorithm runtime fields.

This provenance does **not** guarantee a bit-identical stochastic rerun. CUDA
kernels, library versions, runtime scheduling, or hardware can still change a
sample even when the model revision, prompt, sampler settings, and seed are
fixed. Accordingly, the saved v2 outputs are one-shot samples: the stored
generated token IDs, prompt hashes, revisions, checkpoints, and checksums are
the authoritative artifacts. A rerun is a replication or sensitivity check,
not a replacement chosen after observing the original result.

## Tool requests and action results

The runner recognizes explicit Qwen `<tool_call>...</tool_call>` blocks. Every
parsed call has `status: requested`. Plain endpoint text is not a tool call, and
a requested call is not an executed action.

Only the safe simulated executor may create an action record with an explicit
result such as `executed` or `blocked`. This boundary controls the black-box
outcome label.

## Activation and cost artifacts

Each paid turn records:

- exact rendered input, input token IDs, and prompt hash;
- raw generation, generated token IDs, thinking text, and visible final text;
- model and tokenizer revisions;
- requested activation layers and saved artifact path;
- input, generated, thinking, and final-output token counts;
- measured H200 method time and an estimated per-turn GPU cost.

The cost ledger is stored under:

```text
costs/<trajectory_id>/<thinking_mode>/step_<step_index>/<modal_input_id>.json
```

The estimate does not replace Modal billing. Modal exposes two different
authoritative values:

- the granular **metered resource cost** for each App before credits and
  reservations;
- the workspace **billed cost** after cycle-level credits and adjustments.

Generate a tier-specific Scenario 1 reconciliation after the final billed hour
has closed and Modal has had a few minutes to collect usage:

```bash
uv run --extra modal python \
  scripts/02_model_execution/08_reconcile_modal_billing.py \
  --start 2026-07-28T00:00:00Z \
  --billing-cycle 2026-07 \
  --analysis-tier definitive \
  --output-json results/scenario1/2026-07-31_modal_billing_reconciliation.json \
  --output-csv results/scenario1/2026-07-31_modal_billing_reconciliation.csv \
  --output-md results/scenario1/2026-07-31_modal_billing_reconciliation.md
```

The script requires exactly one `analysis_tier`, filters authoritative hourly
rows using both that tag and `project=spec-gap`, and filters local checkpoints
using the same saved tier. It rejects mid-App tier changes and never pools
`exploratory` and `definitive` rows. The workspace billed total is not allocated
to individual Apps because Modal applies credits at the workspace cycle level.

Paid single-trajectory and batch entry points also attach the domain,
generation protocol, run kind, and analysis tier to each new App. The safe
default is `exploratory`; a frozen matrix must opt in with
`--analysis-tier definitive` before compute starts. The tier is also persisted
through the request, result, live turn, and local cost metadata so a resumed
checkpoint cannot silently cross tiers.

The checked-in July 31 v1 reconciliation predates analysis-tier tagging. It is
a legacy all-project infrastructure total, not an exploratory or definitive
paper cost table. Untagged historical Apps are excluded from new tier-specific
reconciliations.

The per-protocol CSV/JSON ledger applies the same isolation rule: every row
carries its saved tier and mixed-tier inputs are rejected. It reports declared
trajectory-count completeness when supplied, but deliberately leaves paper
cost-table eligibility to the frozen analysis manifest and review. Historical
records without a saved tier remain auditable as `unclassified`.

## Complete trajectory runner

Runbook stage `S09` connects the one-turn backend to the sequential Scenario 1
planner-worker-worker2-executor orchestrator. It forwards only visible final
content and uses a no-network simulated executor for tool requests.

Validate a complete request plan without a GPU:

```bash
modal run \
  scripts/02_model_execution/04_run_scenario1_live.py::run_scenario1_trajectory \
  --condition-id 2-hop \
  --treatment clean \
  --thinking-mode off \
  --action validate
```

A paid trajectory additionally requires
`--action run --confirm-paid-run RUN_H200_TRAJECTORY`. Run one complete
trajectory and validate its saved v2 JSON before launching the full 56-turn
two-mode batch.

Use the resumable batch entry point after that first trajectory passes:

```bash
modal run \
  scripts/02_model_execution/05_run_scenario1_batch.py::run_scenario1_batch \
  --action validate

modal run \
  scripts/02_model_execution/05_run_scenario1_batch.py::run_scenario1_batch \
  --action run \
  --analysis-tier definitive \
  --confirm-paid-run RUN_H200_BATCH
```

The batch skips existing valid trajectories, checkpoints every model turn, and
keeps the remaining sequential calls inside one Modal app. This avoids loading
Qwen3-32B again for every trajectory. Add `--max-new-trajectories 1` to bound a
paid run while checking a new environment.

New live JSON, model-turn checkpoints, and activation artifacts are physically
namespaced by `analysis_tier`, so an exploratory run and a definitive rerun can
coexist without overwriting or blocking each other. Legacy artifacts remain in
their historical tierless paths and are treated as `unclassified`.

The raw-poison Worker1 result includes the injection's exact character and
token spans in the rendered Qwen prompt. The saved tokenizer revision and
prompt hash make that alignment reproducible for later activation analysis.

The activation artifact for each turn contains all 64 layers at up to three
named token positions: `last_input_token`, `last_reasoning_token` for thinking
on, and `last_visible_answer_token`. The original `activations` tensor remains
the primary last-generated-token view for downstream compatibility. The JSON
stores the position metadata and one artifact checksum; the `.pt` file stores
the tensors. `last_input_token` is extracted in a separate prompt-only forward
pass so its strict negative control does not depend on the length or content of
the generated continuation. Reasoning and visible-answer checkpoints use the
generated prefix. Both the JSON and artifact record these scopes in
`checkpoint_forward_scopes`.

## Repairing legacy input checkpoints

The initial 16-run batch was saved before `last_input_token` used its own
prompt-only forward pass. The generated responses and generated-token
activations do not need to be rerun. The repair command reads each saved input
token sequence, computes only the final prompt-token activation, and replaces
only that tensor.

First inspect the plan. This is local validation and starts no remote method or
GPU:

```bash
modal run \
  scripts/02_model_execution/06_repair_prompt_activations.py::repair_prompt_activations \
  --action validate \
  --scope smoke_pair
```

The first paid check repairs only the clean and injected thinking-off planners
from one matched 2-hop group:

```bash
modal run \
  scripts/02_model_execution/06_repair_prompt_activations.py::repair_prompt_activations \
  --action run \
  --scope smoke_pair \
  --confirm-paid-run RUN_H200_ACTIVATION_REPAIR_SMOKE
```

The command stops with an error unless those two records have the same prompt
hash, the same input token IDs, and bitwise-identical `last_input_token`
activations at all 64 layers. After that check passes, repair the remaining
legacy artifacts:

```bash
modal run \
  scripts/02_model_execution/06_repair_prompt_activations.py::repair_prompt_activations \
  --action run \
  --scope all \
  --confirm-paid-run RUN_H200_ACTIVATION_REPAIR_ALL
```

The repair uses one H200 model container and processes selected artifacts in
sequence. It does not call generation, change a generated token, or replace a
reasoning or visible-answer activation. Each original artifact is retained at:

```text
activation_backups/prompt_only_last_input_v1/activations/<trajectory_id>/<thinking_mode>/step_<step_index>.pt
```

The updated checkpoint JSON and live trajectory JSON record the repair method,
old and new checksums, forward-pass scope, model revision, prompt and token
hashes, and a cost record with `generated_tokens: 0`. A retry is safe: a remote
artifact that was repaired before local metadata was written is detected and
returned without performing the forward pass again.

If a local process stops after writing the per-turn checkpoint but before
writing the matching live trajectory, finish that metadata-only transaction
without starting a GPU:

```bash
modal run \
  scripts/02_model_execution/06_repair_prompt_activations.py::repair_prompt_activations \
  --action reconcile \
  --scope all
```

Reconciliation verifies the repaired artifact, original backup, checksums,
repair provenance, cost record, and unchanged generated checkpoints before it
updates the live trajectory. A full validation also checksum-checks every
planned local artifact, including artifacts already marked complete.

The command writes a compact local run report under
`results/scenario1/activation_repair/`. The smoke report includes the exact
all-layer equality result; the full report lists every repaired turn and its
estimated method cost.
