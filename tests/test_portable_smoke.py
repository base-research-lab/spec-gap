"""Tests for the clean-checkout Scenario 1 smoke command."""

import runpy
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_portable_smoke_test.py"


def _script_namespace() -> dict:
    return runpy.run_path(SCRIPT_PATH, run_name="spec_gap_portable_smoke_test")


def test_local_smoke_covers_every_active_package(tmp_path):
    summary = _script_namespace()["run_local_smoke"](tmp_path)

    assert summary["status"] == "passed"
    assert summary["model_called"] is False
    assert summary["gpu_started"] is False
    assert summary["domain_count"] == 11
    assert summary["trajectory_count"] == 44
    assert summary["schema_validated_trajectory_count"] == 44
    assert summary["modal_request_template_count"] == 308
    assert summary["cohorts"] == [
        {
            "cohort_id": "shared_core",
            "domain_count": 2,
            "trajectory_count": 8,
            "schema_validated_trajectory_count": 8,
            "modal_request_template_count": 56,
            "analysis_tier": "exploratory",
        },
        {
            "cohort_id": "active_fellow_packages",
            "domain_count": 9,
            "trajectory_count": 36,
            "schema_validated_trajectory_count": 36,
            "modal_request_template_count": 252,
            "analysis_tier": "exploratory",
        },
    ]


def test_fellow_discovery_excludes_archived_configs():
    paths = _script_namespace()["active_fellow_registry_paths"]()

    assert len(paths) == 9
    assert all(path.name == "domain_config.json" for path in paths)
    assert all("archive" not in path.parts for path in paths)


def test_local_smoke_refuses_to_overwrite_retained_output(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("do not replace\n")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        _script_namespace()["run_local_smoke"](tmp_path)

    assert sentinel.read_text() == "do not replace\n"


def test_modal_connectivity_explains_missing_authentication(monkeypatch):
    namespace = _script_namespace()
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="Token not found",
        ),
    )

    with pytest.raises(RuntimeError, match="modal setup"):
        namespace["run_modal_connectivity_check"]("modal")


def test_modal_connectivity_never_runs_the_production_app(monkeypatch):
    namespace = _script_namespace()
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="authenticated",
            stderr="",
        )

    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        fake_run,
    )

    assert namespace["run_modal_connectivity_check"]("modal") == {
        "status": "passed",
        "authenticated": True,
        "workspace_access_verified": True,
        "remote_app_started": False,
        "image_build_started": False,
        "model_called": False,
        "gpu_started": False,
    }
    assert calls == [
        ["modal", "token", "info"],
        ["modal", "app", "list", "--json"],
    ]


def test_modal_connectivity_explains_connection_failure(monkeypatch):
    namespace = _script_namespace()
    monkeypatch.setattr(
        namespace["subprocess"],
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="Could not connect to the Modal server",
        ),
    )

    with pytest.raises(RuntimeError, match="Check network access"):
        namespace["run_modal_connectivity_check"]("modal")
