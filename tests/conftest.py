from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def reference(project_root: Path) -> Path:
    return project_root / "artifacts" / "approved_reference"
