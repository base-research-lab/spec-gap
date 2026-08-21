<!--
Purpose: authoritative landing page for the active SPEC-GAP codebase.
Last reorganized: 2026-08-10.
This is the repository's only README. Detailed procedures live in the linked
runbook and reference guides rather than being repeated here.
-->

<h1 align="center">SPEC-GAP</h1>

<p align="center">
  Tracing indirect prompt injection through a multi-agent system with behavioral records and white-box model activations.
</p>

SPEC-GAP studies the gap between visible model behavior and internal model
state. Scenario 1 places an indirect prompt injection inside one retrieved
document and follows its influence through a planner, one or two workers, and
an executor.

This repository contains the controlled inputs, execution and analysis code,
reproducible reporting tools, and frozen nine-domain existing-data analysis.
It does not contain a new Scenario 1 redesign or rerun; that future work belongs
to the research group.

## Current status

- The active model contract is `Qwen/Qwen3-32B` at revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Thinking off is primary; thinking on is a separate sensitivity analysis.
- The frozen 2026-08-06 cohort contains 9 domains, 72 trajectories, 252 model
  turns, and 630 activation checkpoints across all 64 residual-stream layers.
- Those runs predate execution-tier tagging and are labeled `unclassified`,
  never retroactively `definitive`.
- All 36 injected trajectories resisted under the automatic endpoint rule,
  but the two-person behavioral review remains blank and fail closed.
- Worker 1, thinking off, layer 40 reaches mean held-out-domain AUROC 0.889 for
  clean-versus-injected construction. This is not compromise-detection AUROC.

## Quick start

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,modal]"

python scripts/00_repository/00_show_pipeline.py --check
python scripts/run_portable_smoke_test.py
```

The first command validates the canonical `S00`–`S21` run order. The portable
smoke test builds and schema-validates 44 structural trajectories across 11
active domains plus 308 Modal request templates. It uses temporary storage and
starts no model, app, image build, or GPU.

Run the complete test suite with:

```bash
python -m pytest -q
```

Building trajectories from source PDFs also requires Poppler's `pdftotext`;
see the
[construction prerequisite](docs/scenario1/pipeline-runbook.md#s03-s04-only-when-constructing-or-changing-a-package).

Choose only the path needed for the task:

| Goal | Run | Stop point |
| --- | --- | --- |
| Check a checkout or code change | `S00`–`S02` | Portable smoke passes |
| Rebuild public figures | `S00`–`S02`, then `S18` | Reporting bundle passes |
| Add or change a domain package | `S00`–`S04` | Package and schema validation pass |
| Check authorized Modal access | `S05` | Workspace and billing owner are confirmed |
| Execute a new experiment | `S05`–`S11` | Every research and paid gate passes |
| Analyze hydrated run artifacts | `S12`–`S20` | Cohort, tier, policy, and hashes agree |
| Finalize behavioral labels | `S21` | Two human reviews and adjudication are complete |

The [pipeline runbook](docs/scenario1/pipeline-runbook.md) is the only detailed
operating sequence. Existing filename numbers are phase-local; `S00`–`S21` are
the repository-wide order.

![Scenario 1 evaluation pipeline](docs/assets/scenario1_pipeline_overview.png)

## Experiment at a glance

Each domain package contains three clean documents. One is the carrier; the
injected condition inserts one registered payload while preserving matched
retrieval selection. Each package expands to clean/injected 2-hop and 3-hop
records. Running both thinking modes produces eight trajectories and 28 model
turns per package.

| Property | Controlled value |
| --- | --- |
| 2-hop topology | planner → worker_1 → executor |
| 3-hop topology | planner → worker_1 → worker_2 → executor |
| Injection entry point | Worker 1 at both depths |
| Retrieved documents | Three model-facing document views |
| Seed | `0` |
| Primary generation | `enable_thinking=false` |
| Sensitivity generation | `enable_thinking=true` |
| GPU backend | Modal, one H200 per active model container |

Only Worker 1 receives retrieved documents. Downstream agents receive visible
messages, never raw documents or hidden reasoning. The executor uses a
simulated, no-network tool, so it cannot contact the registered endpoint.

## Repository structure

| Location | Responsibility |
| --- | --- |
| `experiments/scenario1/inputs/` | Canonical tasks, documents, injections, and provenance |
| `schemas/scenario1/v2/` | Machine-readable trajectory and event contracts |
| `scripts/00_repository/` | Environment-independent repository checks |
| `scripts/01_scenario_construction/` | Package construction, validation, and source audits |
| `scripts/02_model_execution/` | Guarded Modal execution, repair, and billing |
| `scripts/03_probe_analysis/` | Activation indexing, controls, probes, and depth analysis |
| `scripts/04_reporting/` | Public figures, fixed analysis, robustness, and review packets |
| `scripts/90_runway_reproduction/` | Frozen historical runway reproduction only |
| `src/` | Reusable implementation imported by scripts and tests |
| `results/scenario1/` | Frozen Scenario 1 results and compact evidence |
| `results/runway/` | Historical runway reports and lightweight rerun outputs |
| `results/presentation/` | Generated presentation figures |
| `docs/` | The runbook, technical reference guides, and documentation assets |
| `notebooks/` | Historical exploratory notebooks, not canonical entry points |
| `archive/` | Obsolete designs retained only for provenance |
| `tests/` | Unit, integration, provenance, naming, and reporting checks |

Use `scripts/` to run the pipeline and `src/` to edit reusable logic. Generated
or historical outputs belong under `results/`; documentation belongs under
`docs/`.

## Inputs and artifact boundaries

Active fellow packages use one predictable structure. Worker_1 receives whole
documents at retrieval time; there is no separate retrieval-plan or chunking
artifact:

```text
experiments/scenario1/inputs/fellow_packages_New/<domain>/
├── registry.json
├── <domain>_doc*_clean.pdf
├── <domain>_doc*_inj_{beginning,middle,before_references}.pdf
├── documents/                 # extracted *_clean.txt fixtures
└── attack_styles/<style>/<position>/domain_config.json
```

Each domain's `registry.json` expands to a 3x3 grid of injection style
(`12`, `20`, `28`) x position (`begin`, `middle`, `end`) via
`scripts/01_scenario_construction/12_build_new_grid_styles_nochunk.py`.

The active domains are `aihc`, `convex`, `fin`, `kg`, `macro`, `neuro`,
`petro`, `policy`, and `telecoms`. Historical pre-grid packages are preserved
for provenance under `archive/old_experiment_inputs/fellow_packages/`.

Large raw trajectories and activation tensors are intentionally ignored. Git
tracks canonical inputs, compact evidence, public summaries, checksums, and
claim limitations. Dates and protocol IDs remain in immutable result names or
metadata because they are scientific provenance, not configuration clutter.

## Results and claim boundary

The frozen combined analysis leaves one complete domain out at a time and
keeps related clean, injected, 2-hop, and 3-hop records in one fold. Planner
last-input activations are the exact pre-retrieval negative control. Robustness
checks cover injection style, exposure design, domain residualization, and a
balanced within-domain permutation null.

These results support separability of the saved construction labels. They do
not establish detection of successful behavioral compromise.

Key artifacts:

- [Analysis manifest](results/scenario1/nine_domain_analysis/fixed_layer_analysis/scenario1_nine_domain_2026_08_06_analysis_manifest.json)
- [Cross-domain robustness summary](results/scenario1/nine_domain_analysis/robustness/cross_domain_robustness.md)
- [Planner negative control](results/scenario1/nine_domain_analysis/all_layer_analysis/figures/scenario1_nine_domain_2026_08_06_all_domains_planner_negative_control.png)
- [Worker 1 fixed-layer figure](results/scenario1/nine_domain_analysis/fixed_layer_analysis/figures/all_domains/thinking_off/scenario1_nine_domain_2026_08_06_all_domains_worker1_thinking_off_auroc_by_layer.png)

Rebuild the public reporting bundle without private trajectories or raw tensors:

```bash
python scripts/04_reporting/15_build_reporting_bundle.py
```

Its tracked input is `results/scenario1/reporting_snapshot.json`.

## Remote and human gates

Modal resources and billing belong to the workspace selected by the active
profile; they are not tied to one person's filesystem. Run `S05` before any
remote preparation. Paid H200 stages require explicit confirmation strings and
research-group approval.

Automatic outcomes are not human semantic judgments. Two independent reviewers
must complete and lock the blinded and treatment-aware phases; disagreement or
machine-fact mismatch requires adjudication. Telecom's separate blinded
style-camouflage rating is also pending. AI-generated ratings are not valid
substitutes.

## Reference guides

- [Canonical pipeline runbook](docs/scenario1/pipeline-runbook.md)
- [Domain-package build guide](docs/scenario1/package-build-guide.md)
- [Trajectory schema guide](docs/scenario1/schema.md)
- [Modal execution and billing](docs/modal.md)
- [Historical runway reproduction](docs/runway.md)

## Reproducibility rules

- Keep active filenames stable and descriptive.
- Preserve source, license, protocol, date, and SHA-256 provenance.
- Never overwrite a historical result with a newer run.
- Keep human judgments separate from immutable trajectory files.
- Keep exploratory, definitive, and historical `unclassified` tiers separate.
- Treat requested tool calls as requests, never executed actions.

## License

Repository-authored code and metadata are licensed under [MIT](LICENSE).
Third-party documents retain their original licenses and redistribution terms.
Neuro-specific restrictions are recorded in
[`LICENSE_NOTICE.md`](archive/old_experiment_inputs/fellow_packages/neuro/LICENSE_NOTICE.md)
(the notice predates the `fellow_packages_New` package layout; its per-source
license table still governs `neuro_doc1`–`neuro_doc3` there — the controlled
injection remains applied only to `neuro_doc1`, the CC BY 4.0 source).

## Citation

Citation metadata is recorded in [`CITATION.cff`](CITATION.cff). Also record the
repository commit and exact result artifact paths used for an analysis.
