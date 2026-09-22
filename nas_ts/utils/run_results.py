"""
One machine-readable record per run: <run_dir>/results.json.

A run's outcome otherwise lives only in a log line and a .pt file, which means
comparing dozens of runs involves grepping logs. This writes the numbers plus
everything needed to tell two runs apart -- seeds, split protocol, code version,
config hash -- so an aggregation step can be a pure read.

A failed run gets a record too, with status "failed" and the error. A missing
file then means "never ran", not "crashed", and the two need different responses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger

STATUS_OK = "ok"
STATUS_FAILED = "failed"

RESULTS_FILENAME = "results.json"


def git_commit_info(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Current commit SHA and whether the working tree is dirty.

    A dirty tree means the results cannot be reproduced from the SHA alone, which
    is exactly what a reader needs to know before trusting a number.
    """
    cwd = str(repo_dir) if repo_dir else None
    info: Dict[str, Any] = {"git_sha": None, "git_dirty": None}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if sha.returncode != 0:
            return info
        info["git_sha"] = sha.stdout.strip()

        st = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if st.returncode == 0:
            info["git_dirty"] = bool(st.stdout.strip())
    except Exception as e:  # git absent, not a repo, timeout
        logger.warning(f"[Results] Could not read git info: {e}")
    return info


def config_hash(cfg: dict) -> str:
    """
    Stable hash of the resolved config, so two runs can be checked for "same
    settings?" without diffing nested dicts. Sorted keys make it order-proof.
    """
    canonical = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_safe(value):
    """
    JSON has no NaN or Infinity. Python's encoder emits them anyway, producing a
    file that pandas reads but a strict parser rejects. Non-finite floats become
    null, which every consumer reads as "no value".
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_results_json(path, payload: Dict[str, Any]) -> Path:
    """
    Write atomically: a temp file in the same directory, then os.replace.

    A reader (the suite's resume check, an aggregation run) must never see a
    half-written file, and os.replace is atomic within a filesystem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")

    text = json.dumps(_json_safe(payload), indent=2, sort_keys=False, allow_nan=False)
    tmp.write_text(text + "\n")
    os.replace(tmp, path)
    return path


def build_results(
    *,
    status: str,
    run_name: str,
    config_path: str,
    cfg: dict,
    wall_clock_seconds: float,
    meta: Optional[dict] = None,
    test_metrics: Optional[dict] = None,
    best_genome: Optional[dict] = None,
    best_params: Optional[float] = None,
    n_evaluated: Optional[int] = None,
    error: Optional[str] = None,
    repo_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the record. Every field is optional except the ones that
    identify the run, so a failure can be recorded with whatever is known."""
    meta = meta or {}
    test_metrics = test_metrics or {}
    tab = (cfg.get("data", {}) or {}).get("tabular", {}) or {}

    payload: Dict[str, Any] = {
        "status": status,
        "run_name": run_name,
        "config_path": str(config_path),
        "task_type": (cfg.get("task", {}) or {}).get("task_type"),

        "dataset_path": tab.get("path"),
        # From meta, not the config: this is what the loader actually did.
        "split_mode": meta.get("split_mode", tab.get("split_mode")),
        "val_is_test": meta.get("val_is_test"),
        "scaling": meta.get("scaling"),

        "evo_random_seed": (cfg.get("evo", {}) or {}).get("random_seed"),
        # meta first: it is what the loader used, config may have left it unset.
        "data_split_seed": meta.get("random_seed", tab.get("random_seed")),

        "test_accuracy": test_metrics.get("test_accuracy"),
        "test_macro_accuracy": test_metrics.get("test_macro_accuracy"),
        "test_loss": test_metrics.get("test_loss"),
        "best_val_loss": test_metrics.get("best_val_loss"),

        "best_params": best_params,
        "best_genome": best_genome,
        "n_evaluated": n_evaluated,
        "wall_clock_seconds": round(float(wall_clock_seconds), 3),

        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "n_test": meta.get("n_test"),
        "num_classes": meta.get("num_classes"),

        "config_hash": config_hash(cfg),
        "error": error,
    }
    payload.update(git_commit_info(repo_dir))
    return payload
