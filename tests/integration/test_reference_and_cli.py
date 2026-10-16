from __future__ import annotations

import json
import os
import subprocess
import sys

from pv_thesis.execution import reproduce_canonical, reproduce_raw
from pv_thesis.verification import run_verification


def test_reference_hashes_schemas_and_counts(project_root):
    result = run_verification(project_root)
    assert result["status"] == "PASS"
    assert all(item["status"] == "PASS" for item in result["hash_checks"])
    assert result["aligned_schema"]["duplicate_keys"] == 0


def test_cli_help_works_after_install(project_root):
    result = subprocess.run([sys.executable, "-m", "pv_thesis", "--help"], cwd=project_root, text=True, capture_output=True)
    assert result.returncode == 0
    assert "forecasting" in result.stdout and "doctor" in result.stdout


def test_consolidated_cli_help_pages_work(project_root):
    for arguments in (["verify", "--help"], ["reproduce", "--help"], ["report", "--help"]):
        result = subprocess.run([sys.executable, "-m", "pv_thesis", *arguments], cwd=project_root, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_consolidated_verify_emits_one_json_document(project_root):
    result = subprocess.run(
        [sys.executable, "-m", "pv_thesis", "verify", "--all"],
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert set(payload["branches"]) == {"forecasting", "physical", "parametric"}


def test_canonical_requires_explicit_confirmation(project_root):
    result = reproduce_canonical(project_root, None, confirmed=False)
    assert result["status"] == "NOT TESTED"
    assert "36–120" in result["resource_estimate"]["wall_clock_range"]


def test_raw_is_blocked_without_private_data(project_root, monkeypatch):
    monkeypatch.delenv("PV_THESIS_DATA_DIR", raising=False)
    result = reproduce_raw(project_root, None, confirmed=False)
    assert result["status"] == "BLOCKED"
