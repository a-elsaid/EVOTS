"""
Shared test helpers.

The tabular CSVs are fetched, not committed (data/ is gitignored), so a fresh
clone or a compute node without network has no data/tabular. Tests that read
those files skip with an actionable reason instead of erroring; everything that
does not touch the data keeps running, which is most of the suite.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TABULAR_DIR = REPO_ROOT / "data" / "tabular"

FETCH_HINT = "dataset CSVs not found; run python fetch_datasets.py"


def tabular_csv_path(name: str) -> Path:
    return TABULAR_DIR / f"{name}.csv"


def missing_tabular(*names: str) -> list:
    """Which of these dataset CSVs are absent, if any."""
    return sorted(n for n in names if not tabular_csv_path(n).exists())


def require_tabular(*names: str) -> None:
    """
    Skip the calling test when any named dataset CSV is missing.

    Called inside the test rather than as a module-level skipif so a
    parametrised test skips per dataset: a run with only iris fetched still
    exercises iris.
    """
    missing = missing_tabular(*names)
    if missing:
        pytest.skip(f"{FETCH_HINT} (missing: {', '.join(missing)})")
