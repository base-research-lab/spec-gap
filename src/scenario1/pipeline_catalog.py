"""Canonical repository-wide run order for Scenario 1.

Created: 2026-08-10.
Purpose: give contributors one machine-checked sequence across construction,
Modal execution, analysis, reporting, and external human review.

The numbered script filenames are stable, phase-local entry points. ``S00``
through ``S23`` are the unique repository-wide stage identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


RUNBOOK_PATH = Path("docs/scenario1/pipeline-runbook.md")
TABLE_START = "<!-- BEGIN GENERATED PIPELINE TABLE -->"
TABLE_END = "<!-- END GENERATED PIPELINE TABLE -->"


@dataclass(frozen=True)
class PipelineStage:
    """One ordered, user-facing pipeline stage."""

    stage_id: str
    phase: str
    title: str
    gate: str
    compute: str
    entrypoints: tuple[str, ...]
    prerequisites: tuple[str, ...]
    result: str


PIPELINE_STAGES = (
    PipelineStage(
        "S00",
        "Setup",
        "Install the project environment",
        "all contributors",
        "local only",
        ("pyproject.toml",),
        (),
        "Editable development and Modal dependencies installed",
    ),
    PipelineStage(
        "S01",
        "Setup",
        "Inspect and validate this run order",
        "all contributors",
        "local only",
        ("scripts/00_repository/00_show_pipeline.py",),
        ("S00",),
        "Catalog, entry-point, and runbook consistency check",
    ),
    PipelineStage(
        "S02",
        "Setup",
        "Run the portable repository smoke test",
        "all contributors",
        "local only",
        ("scripts/run_portable_smoke_test.py",),
        ("S00",),
        "44 validated structural trajectories and 308 request templates",
    ),
    PipelineStage(
        "S03",
        "Construction",
        "Generate structural trajectories",
        "package changes",
        "local only",
        ("scripts/01_scenario_construction/01_generate_trajectories.py",),
        ("S02",),
        "Matched clean/injected 2-hop and 3-hop records",
    ),
    PipelineStage(
        "S04",
        "Construction",
        "Validate structural trajectories",
        "package changes",
        "local only",
        ("scripts/01_scenario_construction/02_validate_trajectories.py",),
        ("S03",),
        "Schema and semantic validation pass",
    ),
    PipelineStage(
        "S05",
        "Modal",
        "Verify the selected Modal workspace",
        "authorized workspace",
        "read-only network",
        ("scripts/run_portable_smoke_test.py",),
        ("S02",),
        "Authenticated workspace check with no app or GPU start",
    ),
    PipelineStage(
        "S06",
        "Modal",
        "Cache the pinned model when needed",
        "authorized workspace",
        "remote CPU/storage",
        ("scripts/02_model_execution/03_modal_qwen_runner.py",),
        ("S05",),
        "Pinned Qwen revision stored in the selected workspace",
    ),
    PipelineStage(
        "S07",
        "Modal",
        "Validate one complete trajectory plan",
        "authorized workspace",
        "remote app setup",
        ("scripts/02_model_execution/04_run_scenario1_live.py",),
        ("S04", "S05"),
        "Exact no-model trajectory preview",
    ),
    PipelineStage(
        "S08",
        "Modal",
        "Run one bounded exploratory trajectory",
        "lab-owned",
        "paid H200",
        ("scripts/02_model_execution/04_run_scenario1_live.py",),
        ("S07",),
        "Saved exploratory trajectory, checkpoints, activations, and costs",
    ),
    PipelineStage(
        "S09",
        "Modal",
        "Run or resume the approved definitive matrix",
        "lab-owned",
        "paid H200",
        ("scripts/02_model_execution/05_run_scenario1_batch.py",),
        ("S08",),
        "Complete tier-isolated matrix for the explicitly selected cohort",
    ),
    PipelineStage(
        "S10",
        "Accounting",
        "Summarize the execution protocol ledger",
        "lab-owned",
        "local only",
        ("scripts/02_model_execution/07_summarize_scenario1_protocol.py",),
        ("S09",),
        "Protocol-specific trajectory and cost ledger",
    ),
    PipelineStage(
        "S11",
        "Accounting",
        "Reconcile Modal billing after data settles",
        "authorized workspace",
        "read-only network",
        ("scripts/02_model_execution/08_reconcile_modal_billing.py",),
        ("S09",),
        "Tier-specific metered and billed cost reconciliation",
    ),
    PipelineStage(
        "S12",
        "Analysis",
        "Build the activation index",
        "lab-owned",
        "local artifacts",
        ("scripts/03_probe_analysis/07_build_activation_index.py",),
        ("S09",),
        "One-tier activation index with policy and checksum provenance",
    ),
    PipelineStage(
        "S13",
        "Analysis",
        "Scan layers and run activation controls",
        "lab-owned",
        "local CPU",
        ("scripts/03_probe_analysis/08_scan_activation_layers.py",),
        ("S12",),
        "Exploratory all-layer scan and paired negative-control audit",
    ),
    PipelineStage(
        "S14",
        "Analysis",
        "Plot the exploratory layer scan",
        "optional",
        "local CPU",
        ("scripts/03_probe_analysis/09_plot_layer_scan.py",),
        ("S13",),
        "Guarded diagnostic layer-scan figures",
    ),
    PipelineStage(
        "S15",
        "Analysis",
        "Score group-held-out probes",
        "lab-owned",
        "local CPU",
        ("scripts/03_probe_analysis/10_score_baseline_probes.py",),
        ("S12",),
        "Per-step Goldowsky-Dill and LAT scores",
    ),
    PipelineStage(
        "S16",
        "Analysis",
        "Analyze depth degradation",
        "lab-owned",
        "local CPU",
        ("scripts/03_probe_analysis/11_analyze_depth_degradation.py",),
        ("S15",),
        "Depth, calibration, and temporal analysis artifacts",
    ),
    PipelineStage(
        "S17",
        "Analysis",
        "Build the reporting snapshot and analysis figures",
        "lab-owned",
        "local CPU",
        ("scripts/03_probe_analysis/12_plot_probe_analysis.py",),
        ("S16",),
        "Compact hash-bound snapshot, figures, tables, and manifest",
    ),
    PipelineStage(
        "S18",
        "Reporting",
        "Rebuild the public reporting bundle",
        "all contributors",
        "local CPU",
        ("scripts/04_reporting/15_build_reporting_bundle.py",),
        ("S00",),
        "Public figures rebuilt from the tracked compact snapshot",
    ),
    PipelineStage(
        "S19",
        "Reporting",
        "Build the fixed-layer analysis",
        "lab-owned",
        "local CPU",
        ("scripts/04_reporting/16_build_fixed_layer_analysis.py",),
        ("S12", "S15", "S16"),
        "Prespecified fixed-layer tables, figures, and manifest",
    ),
    PipelineStage(
        "S20",
        "Reporting",
        "Build cross-domain robustness checks",
        "lab-owned",
        "local CPU",
        ("scripts/04_reporting/17_build_cross_domain_robustness.py",),
        ("S12", "S15", "S19"),
        "Ablations, paired deltas, nulls, and residualization checks",
    ),
    PipelineStage(
        "S21",
        "Human review",
        "Build and complete the two-reviewer outcome packet",
        "external humans",
        "human judgment",
        ("scripts/04_reporting/18_build_cross_domain_human_review.py",),
        ("S12",),
        "Hash-locked reviews and adjudication, or an explicit pending gate",
    ),
)


ALLOWED_GATES = {
    "all contributors",
    "package changes",
    "authorized workspace",
    "lab-owned",
    "optional",
    "external humans",
}
ALLOWED_COMPUTE = {
    "local only",
    "read-only network",
    "remote CPU/storage",
    "remote app setup",
    "paid H200",
    "local artifacts",
    "local CPU",
    "human judgment",
}


def pipeline_catalog_rows() -> list[dict[str, object]]:
    """Return JSON-serializable catalog rows in canonical order."""

    rows = []
    for stage in PIPELINE_STAGES:
        row = asdict(stage)
        row["entrypoints"] = list(stage.entrypoints)
        row["prerequisites"] = list(stage.prerequisites)
        rows.append(row)
    return rows


def render_markdown_table() -> str:
    """Render the table embedded verbatim in the runbook."""

    lines = [
        "| Stage | Phase | Who runs it | Compute boundary | Entrypoint | Result |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for stage in PIPELINE_STAGES:
        entrypoints = "<br>".join(f"`{path}`" for path in stage.entrypoints)
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{stage.stage_id}`",
                    stage.phase,
                    stage.gate,
                    stage.compute,
                    entrypoints,
                    stage.result,
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_text_catalog() -> str:
    """Render a compact terminal-oriented run order."""

    lines = [
        "SPEC-GAP canonical run order",
        f"Detailed commands: {RUNBOOK_PATH.as_posix()}",
        "",
    ]
    for stage in PIPELINE_STAGES:
        lines.append(f"{stage.stage_id}  {stage.title} [{stage.gate}; {stage.compute}]")
        lines.append(f"     -> {stage.result}")
    return "\n".join(lines) + "\n"


def validate_pipeline_catalog(project_root: Path) -> None:
    """Fail if ordering metadata or a declared entry point is invalid."""

    expected_ids = [f"S{index:02d}" for index in range(len(PIPELINE_STAGES))]
    actual_ids = [stage.stage_id for stage in PIPELINE_STAGES]
    if actual_ids != expected_ids:
        raise ValueError(
            "Pipeline stage IDs must be unique, contiguous, and ordered: "
            f"expected {expected_ids}, got {actual_ids}."
        )

    known_ids: set[str] = set()
    root = project_root.resolve()
    for stage in PIPELINE_STAGES:
        if stage.gate not in ALLOWED_GATES:
            raise ValueError(f"{stage.stage_id} has unknown gate {stage.gate!r}.")
        if stage.compute not in ALLOWED_COMPUTE:
            raise ValueError(
                f"{stage.stage_id} has unknown compute boundary {stage.compute!r}."
            )
        if not stage.title.strip() or not stage.result.strip():
            raise ValueError(f"{stage.stage_id} has blank user-facing metadata.")
        if not stage.entrypoints:
            raise ValueError(f"{stage.stage_id} has no entry point.")
        for dependency in stage.prerequisites:
            if dependency not in known_ids:
                raise ValueError(
                    f"{stage.stage_id} depends on missing or later stage {dependency}."
                )
        for relative_name in stage.entrypoints:
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"{stage.stage_id} entry point must be repository-relative: "
                    f"{relative_name!r}."
                )
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"{stage.stage_id} entry point escapes the repository: "
                    f"{relative_name!r}."
                ) from error
            if not resolved.is_file():
                raise ValueError(
                    f"{stage.stage_id} entry point does not exist: {relative_name!r}."
                )
        known_ids.add(stage.stage_id)


def validate_pipeline_runbook(project_root: Path) -> None:
    """Fail if the generated runbook table has drifted from this catalog."""

    runbook = project_root / RUNBOOK_PATH
    text = runbook.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(TABLE_START) + r"\n(.*?)" + re.escape(TABLE_END),
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError("The pipeline runbook is missing its generated table markers.")
    if match.group(1) != render_markdown_table():
        raise ValueError(
            "The pipeline runbook table has drifted. Regenerate it with "
            "scripts/00_repository/00_show_pipeline.py --format markdown."
        )


def validate_repository_pipeline(project_root: Path) -> None:
    """Validate both the catalog and its checked-in runbook projection."""

    validate_pipeline_catalog(project_root)
    validate_pipeline_runbook(project_root)
