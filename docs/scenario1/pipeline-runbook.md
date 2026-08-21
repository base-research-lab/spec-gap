<!--
Purpose: canonical, ordered operating guide for the active Scenario 1 pipeline.
Created: 2026-08-10.
The S00-S21 table is generated from src/scenario1/pipeline_catalog.py.
-->

# Scenario 1 pipeline runbook

This is the one ordered operating guide for the active pipeline. Run every
command from the repository root. Use `scripts/` as command entry points and
`src/` when reading or changing reusable implementation code.

Do not run every numbered file simply because it exists. Filename numbers are
stable, phase-local names; `S00` through `S21` below are the unique
repository-wide order. Package-specific evidence builders, legacy repairs, and
the `90_` runway reproduction are side paths, not missing main stages.

## Choose the shortest correct path

```text
S00 install -> S01 inspect order -> S02 portable smoke
                                      |
                                      +-> public checkout: S18 reporting rebuild
                                      +-> package edit: S03 -> S04
                                      +-> new lab run: S05 -> S06 -> S07 -> S08 -> S09
                                                                  |       |
                                                                  |       +-> S10/S11 accounting
                                                                  |       +-> S12-S20 analysis/reporting
                                                                  |       +-> S21 human review
                                                                  +-> stop unless the one-run gate passes
```

- A contributor checking the repository stops after `S02`.
- A contributor rebuilding public figures from tracked compact data jumps from
  `S02` to `S18`; private trajectories and activation tensors are not needed.
- A package author runs `S03` through `S04`, then stops unless the research
  group has approved new model execution.
- The research group owns `S08` onward for any new Scenario 1 run. Repository
  cleanup does not authorize a redesign, rerun, or definitive paper claim.

## Canonical stage catalog

Print the same catalog in a terminal, or fail if its paths or this table have
drifted:

```bash
python scripts/00_repository/00_show_pipeline.py
python scripts/00_repository/00_show_pipeline.py --check
```

<!-- BEGIN GENERATED PIPELINE TABLE -->
| Stage | Phase | Who runs it | Compute boundary | Entrypoint | Result |
| --- | --- | --- | --- | --- | --- |
| `S00` | Setup | all contributors | local only | `pyproject.toml` | Editable development and Modal dependencies installed |
| `S01` | Setup | all contributors | local only | `scripts/00_repository/00_show_pipeline.py` | Catalog, entry-point, and runbook consistency check |
| `S02` | Setup | all contributors | local only | `scripts/run_portable_smoke_test.py` | 44 validated structural trajectories and 308 request templates |
| `S03` | Construction | package changes | local only | `scripts/01_scenario_construction/01_generate_trajectories.py` | Matched clean/injected 2-hop and 3-hop records |
| `S04` | Construction | package changes | local only | `scripts/01_scenario_construction/02_validate_trajectories.py` | Schema and semantic validation pass |
| `S05` | Modal | authorized workspace | read-only network | `scripts/run_portable_smoke_test.py` | Authenticated workspace check with no app or GPU start |
| `S06` | Modal | authorized workspace | remote CPU/storage | `scripts/02_model_execution/03_modal_qwen_runner.py` | Pinned Qwen revision stored in the selected workspace |
| `S07` | Modal | authorized workspace | remote app setup | `scripts/02_model_execution/04_run_scenario1_live.py` | Exact no-model trajectory preview |
| `S08` | Modal | lab-owned | paid H200 | `scripts/02_model_execution/04_run_scenario1_live.py` | Saved exploratory trajectory, checkpoints, activations, and costs |
| `S09` | Modal | lab-owned | paid H200 | `scripts/02_model_execution/05_run_scenario1_batch.py` | Complete tier-isolated matrix for the explicitly selected cohort |
| `S10` | Accounting | lab-owned | local only | `scripts/02_model_execution/07_summarize_scenario1_protocol.py` | Protocol-specific trajectory and cost ledger |
| `S11` | Accounting | authorized workspace | read-only network | `scripts/02_model_execution/08_reconcile_modal_billing.py` | Tier-specific metered and billed cost reconciliation |
| `S12` | Analysis | lab-owned | local artifacts | `scripts/03_probe_analysis/07_build_activation_index.py` | One-tier activation index with policy and checksum provenance |
| `S13` | Analysis | lab-owned | local CPU | `scripts/03_probe_analysis/08_scan_activation_layers.py` | Exploratory all-layer scan and paired negative-control audit |
| `S14` | Analysis | optional | local CPU | `scripts/03_probe_analysis/09_plot_layer_scan.py` | Guarded diagnostic layer-scan figures |
| `S15` | Analysis | lab-owned | local CPU | `scripts/03_probe_analysis/10_score_baseline_probes.py` | Per-step Goldowsky-Dill and LAT scores |
| `S16` | Analysis | lab-owned | local CPU | `scripts/03_probe_analysis/11_analyze_depth_degradation.py` | Depth, calibration, and temporal analysis artifacts |
| `S17` | Analysis | lab-owned | local CPU | `scripts/03_probe_analysis/12_plot_probe_analysis.py` | Compact hash-bound snapshot, figures, tables, and manifest |
| `S18` | Reporting | all contributors | local CPU | `scripts/04_reporting/15_build_reporting_bundle.py` | Public figures rebuilt from the tracked compact snapshot |
| `S19` | Reporting | lab-owned | local CPU | `scripts/04_reporting/16_build_fixed_layer_analysis.py` | Prespecified fixed-layer tables, figures, and manifest |
| `S20` | Reporting | lab-owned | local CPU | `scripts/04_reporting/17_build_cross_domain_robustness.py` | Ablations, paired deltas, nulls, and residualization checks |
| `S21` | Human review | external humans | human judgment | `scripts/04_reporting/18_build_cross_domain_human_review.py` | Hash-locked reviews and adjudication, or an explicit pending gate |
<!-- END GENERATED PIPELINE TABLE -->

`S06` is conditional: reuse the pinned model cache when it already exists in
the selected workspace. `S11` can run in parallel with analysis after Modal's
billing data settles. `S14` is diagnostic and does not select the
prespecified fixed layer.

## S00-S02: every checkout starts here

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,modal]"

python scripts/00_repository/00_show_pipeline.py --check
python scripts/run_portable_smoke_test.py
```

The smoke command is checkout-relative, writes only to temporary storage by
default, validates all 11 active domains, and starts no network request, Modal
app, model, or GPU. If it fails, stop here.

To retain its generated dry-run records, provide a new or empty directory:

```bash
python scripts/run_portable_smoke_test.py \
  --output-root PATH/TO/NEW_OR_EMPTY_SMOKE_DIRECTORY
```

The command refuses to overwrite a nonempty path.

## S03-S04: only when constructing or changing a package

Worker_1 receives whole documents. There is no retrieval plan, chunking, or
ranking step: a domain package is its clean and injected source PDFs plus one
`registry.json` (or, for the style x position grid, one
`fellow_packages_New/<domain>/registry.json`). Do not regenerate an existing
active package's structural trajectories merely to inspect the repository.

The canonical source-PDF extraction calls `pdftotext -raw`. That executable
comes from Poppler and is outside the Python environment created at `S00`.
Install it with the operating system's package manager when needed (for
example, `brew install poppler` on macOS or `sudo apt-get install
poppler-utils` on Debian/Ubuntu). On Windows, install an approved Poppler
distribution, add its `bin` directory to `PATH`, and use the PowerShell check
shown below.

The command examples in this guide otherwise use a POSIX-compatible shell.
Verify the executable and version before `S03`:

```bash
pdftotext -v
command -v pdftotext  # optional POSIX location check
```

PowerShell equivalent:

```powershell
pdftotext -v
(Get-Command pdftotext).Source
```

Stop if the command is absent. A Poppler-version extraction difference will
fail the tracked text/hash checks rather than silently changing a document.

For the style x position grid domains, build and validate every cell in one
step:

```bash
python scripts/01_scenario_construction/12_build_new_grid_styles_nochunk.py \
  --domains DOMAIN_OR_COMMA_LIST
```

This writes each cell's `domain_config.json` under
`fellow_packages_New/<domain>/attack_styles/<style>/<position>/` and, unless
`--skip-pipeline` is passed, internally runs `S03` and `S04` (below) against
every cell.

For a one-off or non-grid registry, generate structural records into an
isolated directory, then validate every record before any Modal step:

```bash
python scripts/01_scenario_construction/01_generate_trajectories.py \
  --mode dry_run \
  --registry PATH/TO/domain_config.json \
  --out PATH/TO/NEW_STRUCTURAL_OUTPUT

python scripts/01_scenario_construction/02_validate_trajectories.py \
  PATH/TO/NEW_STRUCTURAL_OUTPUT/*.json
```

Stop if the single-insertion clean/injected diff, schema validation, or
source/license review fails.

## S05-S09: guarded Modal execution

These stages use whichever workspace the active contributor profile selects;
they are not tied to one person's laptop or former lab membership.

If no valid Modal profile exists yet, complete one-time onboarding first:

```bash
modal setup
```

`modal setup` opens the authentication flow and writes local profile/token
configuration. It starts no app or compute, but it is not part of the read-only
`S05` verification.

Once credentials exist, confirm the workspace, inspect its current billing
summary, and run the read-only access check:

```bash
modal profile current
modal token info
modal billing summary --json
python scripts/run_portable_smoke_test.py --check-modal
```

The optional Modal dependency is pinned to 1.5.3 or newer because
`modal billing summary` first became available in that release line.

If the workspace is wrong, access is missing, or the budget is unknown, stop.
Do not infer that another workspace's model Volume or credits are available.

If the pinned revision is absent from this workspace, cache it without a GPU:

```bash
modal run scripts/02_model_execution/03_modal_qwen_runner.py \
  --request-path tests/fixtures/qwen_agent_turn_request.json \
  --action download \
  --model-revision 9216db5781bf21249d130ec9da846c4624c16137
```

This uses remote CPU, memory, network, and Volume storage. Validate one real
trajectory plan next; this enters Modal's app lifecycle and may prepare its
image/resources, but it does not call the model or allocate an H200:

```bash
modal run \
  scripts/02_model_execution/04_run_scenario1_live.py::run_scenario1_trajectory \
  --registry-path PATH/TO/domain_config.json \
  --condition-id 2-hop \
  --treatment clean \
  --thinking-mode off \
  --analysis-tier exploratory \
  --action validate
```

Only the research group should authorize the paid gate. The first paid run is
one exploratory trajectory, never a partially labeled definitive matrix:

```bash
modal run \
  scripts/02_model_execution/04_run_scenario1_live.py::run_scenario1_trajectory \
  --registry-path PATH/TO/domain_config.json \
  --condition-id 2-hop \
  --treatment clean \
  --thinking-mode off \
  --analysis-tier exploratory \
  --action run \
  --confirm-paid-run RUN_H200_TRAJECTORY
```

Review and validate the saved trajectory, checkpoints, activation checksums,
tool/action records, and costs. Only after that gate passes may the group
validate and run its explicitly enumerated definitive cohort:

```bash
modal run \
  scripts/02_model_execution/05_run_scenario1_batch.py::run_scenario1_batch \
  --registry-paths PATH/TO/domain_config.json,PATH/TO/ANOTHER/domain_config.json \
  --thinking-modes off,on \
  --analysis-tier definitive \
  --action validate

modal run \
  scripts/02_model_execution/05_run_scenario1_batch.py::run_scenario1_batch \
  --registry-paths PATH/TO/domain_config.json,PATH/TO/ANOTHER/domain_config.json \
  --thinking-modes off,on \
  --analysis-tier definitive \
  --action run \
  --confirm-paid-run RUN_H200_BATCH
```

`--domains` or `--registry-paths` is required; there is no default cohort.
Record the approved cohort, expected trajectory count, protocol ID, tier,
workspace, and cost ceiling before the paid command.

## S10-S11: execution accounting

Build a tier-specific protocol ledger immediately after the matrix validates:

```bash
python scripts/02_model_execution/07_summarize_scenario1_protocol.py \
  --trajectory-root experiments/scenario1/trajectories/live \
  --generation-protocol-id PROTOCOL_ID \
  --analysis-tier ANALYSIS_TIER \
  --expected-trajectory-count EXPECTED_COUNT \
  --output-csv PATH/TO/protocol_ledger.csv \
  --output-json PATH/TO/protocol_ledger.json
```

After Modal's final billed interval has closed, reconcile the same tier:

```bash
python scripts/02_model_execution/08_reconcile_modal_billing.py \
  --start ISO_8601_START \
  --end ISO_8601_END \
  --billing-cycle YYYY-MM \
  --analysis-tier ANALYSIS_TIER \
  --output-json PATH/TO/billing.json \
  --output-csv PATH/TO/billing.csv \
  --output-md PATH/TO/billing.md
```

Local per-turn estimates are not the invoice. Preserve both Modal's metered
resource cost and billed cost after credits/adjustments.

## S12-S17: analysis after artifact hydration

Hydrate the exact raw trajectories and activation tensors first. The paths and
hashes must match the run ledger; never silently mix `exploratory`,
`definitive`, and historical `unclassified` inputs.

```bash
python scripts/03_probe_analysis/07_build_activation_index.py \
  --trajectory-root experiments/scenario1/trajectories/live \
  --artifact-root PATH/TO/HYDRATED_ARTIFACT_ROOT \
  --analysis-tier ANALYSIS_TIER \
  --output PATH/TO/activation_index.jsonl \
  --summary-output PATH/TO/activation_index_summary.json \
  --require-local \
  --verify-checksums

python scripts/03_probe_analysis/08_scan_activation_layers.py \
  --index PATH/TO/activation_index.jsonl \
  --output-json PATH/TO/layer_scan.json \
  --output-csv PATH/TO/layer_scan.csv \
  --control-output-json PATH/TO/activation_controls.json \
  --control-output-csv PATH/TO/activation_control_pairs.csv

python scripts/03_probe_analysis/09_plot_layer_scan.py \
  --input PATH/TO/layer_scan.json \
  --output-dir PATH/TO/DIAGNOSTIC_FIGURES

python scripts/03_probe_analysis/10_score_baseline_probes.py \
  --index PATH/TO/activation_index.jsonl \
  --manifest experiments/scenario1/manifest.json \
  --output-jsonl PATH/TO/per_step_probe_scores.jsonl \
  --output-summary PATH/TO/per_step_probe_scores_summary.json

python scripts/03_probe_analysis/11_analyze_depth_degradation.py \
  PATH/TO/per_step_probe_scores.jsonl \
  --experiment-id EXPERIMENT_ID \
  --layers 32,40,48 \
  --output-json PATH/TO/depth_degradation.json \
  --output-csv PATH/TO/depth_degradation.csv \
  --temporal-output-jsonl PATH/TO/temporal_divergence.jsonl

python scripts/03_probe_analysis/11_analyze_depth_degradation.py \
  PATH/TO/per_step_probe_scores.jsonl \
  --experiment-id EXPERIMENT_ID-all-layers \
  --layers all \
  --output-json PATH/TO/all_layer_descriptive.json

python scripts/03_probe_analysis/12_plot_probe_analysis.py \
  --reference-result PATH/TO/depth_degradation.json \
  --all-layer-result PATH/TO/all_layer_descriptive.json \
  --per-step-scores PATH/TO/per_step_probe_scores.jsonl \
  --analysis-tier ANALYSIS_TIER \
  --snapshot-output PATH/TO/reporting_snapshot.json \
  --figure-dir PATH/TO/FIGURES \
  --analysis-dir PATH/TO/FINAL_ANALYSIS
```

`S13` is an exploratory scan. `S14` visualizes that scan; it does not choose a
paper layer. Prespecified fixed-layer work begins at `S19`.

## S18-S21: reporting and human gates

Anyone can rebuild the public bundle from the tracked compact snapshot:

```bash
python scripts/04_reporting/15_build_reporting_bundle.py
```

For a newly approved analysis, build the fixed-layer and robustness artifacts
from the exact same index, scores, depth result, policy, and design manifest:

```bash
python scripts/04_reporting/16_build_fixed_layer_analysis.py \
  --scores PATH/TO/per_step_probe_scores.jsonl \
  --depth-result PATH/TO/depth_degradation.json \
  --activation-index PATH/TO/activation_index.jsonl \
  --output-dir PATH/TO/FIXED_LAYER_OUTPUT \
  --stem EXPERIMENT_ID

python scripts/04_reporting/17_build_cross_domain_robustness.py \
  --activation-index PATH/TO/activation_index.jsonl \
  --scores PATH/TO/per_step_probe_scores.jsonl \
  --design-manifest experiments/scenario1/manifest.json \
  --output-dir PATH/TO/ROBUSTNESS_OUTPUT
```

Build the blinded packet only from its hash-bound source roots and machine
evidence. Keep reviewer forms blank in Git:

```bash
python scripts/04_reporting/18_build_cross_domain_human_review.py \
  --activation-index PATH/TO/activation_index.jsonl \
  --source-root DOMAIN=PATH/TO/DOMAIN_TRAJECTORIES \
  --policy-language-audit PATH/TO/policy_request_language_audit.json \
  --policy-pdf-audit PATH/TO/policy_pdf_pair_audit.json \
  --telecom-pdf-audit PATH/TO/telecom_pdf_pair_audit.json \
  --telecom-style-review PATH/TO/telecom_style_review.json \
  --output-dir PATH/TO/NEW_OR_EMPTY_HUMAN_REVIEW_BUNDLE
```

Repeat `--source-root` for every domain in the declared cohort. The four audit
paths shown above are required build bindings, not optional examples. Build
mode refuses any nonempty output directory so it cannot erase human judgments.
Never rebuild a completed bundle; use validation mode only. After two real
reviewers complete and lock both phases, validate the bundle:

```bash
python scripts/04_reporting/18_build_cross_domain_human_review.py \
  --validate-completed-review-dir PATH/TO/HUMAN_REVIEW_BUNDLE
```

An incomplete or disagreeing review stays pending and blocks the associated
human outcome claim. AI-generated ratings are not substitutes for reviewers.

## Side paths and ownership

The following are intentionally outside the main line:

- `scripts/01_scenario_construction/05_*` through `07_*` rebuild
  package-specific language, result-evidence, PDF, or style audits.
- `scripts/02_model_execution/06_repair_prompt_activations.py` repairs a
  documented legacy checkpoint defect; never run it on a normal new matrix.
- `scripts/04_reporting/13_plot_pipeline_overview.py` and
  `14_plot_investor_figures.py` are called by the `S18` bundle wrapper.
- `scripts/90_runway_reproduction/` is frozen historical reproduction and is
  not a prerequisite for active Scenario 1 work.
- A combined natural-text attack, mechanism axis, arbitrary-target tool, or
  other Scenario 1 redesign is future research-group work, not a cleanup step.

## Stop conditions

Stop rather than working around any of these conditions:

- `S01` reports catalog/runbook drift or `S02` fails locally.
- A source, license, tokenizer, or context hash does not match.
- The active Modal workspace, billing owner, or spend ceiling is unclear.
- The no-model preview or single exploratory trajectory has not passed review.
- A purported definitive cohort is partial, mixed-tier, selectively rerun, or
  not explicitly enumerated.
- Trajectories, tensors, scores, depth results, and paper policy select
  different cohorts or tiers.
- Required human review fields or adjudication are incomplete.
