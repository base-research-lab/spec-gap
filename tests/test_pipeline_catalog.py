"""Guard the repository-wide Scenario 1 run order and documentation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.scenario1.pipeline_catalog import (
    PIPELINE_STAGES,
    pipeline_catalog_rows,
    render_markdown_table,
    validate_repository_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/00_repository/00_show_pipeline.py"
RUNBOOK = PROJECT_ROOT / "docs/scenario1/pipeline-runbook.md"


def test_pipeline_catalog_is_contiguous_and_repository_bound() -> None:
    validate_repository_pipeline(PROJECT_ROOT)

    assert [stage.stage_id for stage in PIPELINE_STAGES] == [
        f"S{index:02d}" for index in range(22)
    ]
    assert PIPELINE_STAGES[0].title == "Install the project environment"
    assert PIPELINE_STAGES[-1].gate == "external humans"


def test_pipeline_cli_is_checkout_relative_and_machine_readable(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json", "--check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == pipeline_catalog_rows()
    assert completed.stderr == "Pipeline catalog check passed.\n"


def test_runbook_contains_the_exact_generated_catalog_table() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert render_markdown_table() in text
    assert "Do not run every numbered file" in text
    assert "research group owns" in text
    for required_flag in (
        "--policy-language-audit",
        "--policy-pdf-audit",
        "--telecom-pdf-audit",
        "--telecom-style-review",
    ):
        assert required_flag in text
    assert "pdftotext -v" in text
    assert "Get-Command pdftotext" in text
    assert "outside the Python environment" in text
    assert "NEW_OR_EMPTY_HUMAN_REVIEW_BUNDLE" in text
    assert "Never rebuild a completed bundle" in text


def test_modal_setup_is_separate_from_read_only_workspace_verification() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    verification = text[
        text.index("Once credentials exist") : text.index("If the workspace is wrong")
    ]

    assert "`modal setup`" not in verification
    assert "writes local profile/token" in text
    assert '"modal>=1.5.3,<2"' in (PROJECT_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_readme_points_to_one_start_command_and_one_detailed_runbook() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "python scripts/00_repository/00_show_pipeline.py --check" in readme
    assert "docs/scenario1/pipeline-runbook.md" in readme
    assert "S00" in readme and "S21" in readme
    assert "Poppler's `pdftotext`" in readme
