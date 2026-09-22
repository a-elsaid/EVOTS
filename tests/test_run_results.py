"""
results.json: one machine-readable record per run.

A missing file must mean "never ran" and status "failed" must mean "ran and
broke" -- the suite runner resumes on that distinction, so a crashed run that
left no file would be silently retried forever, and one that left an "ok" file
with no numbers would be silently trusted.
"""

import json
import math
import os
from pathlib import Path

import pytest
import yaml

from nas_ts.utils.run_results import (
    STATUS_FAILED,
    STATUS_OK,
    build_results,
    config_hash,
    git_commit_info,
    write_results_json,
)

CFG = {
    "run": {"name": "r"},
    "task": {"task_type": "classification"},
    "evo": {"random_seed": 7},
    "data": {"tabular": {"path": "data/tabular/iris.csv", "random_seed": 3,
                         "split_mode": "clean"}},
}
META = {"split_mode": "clean", "val_is_test": False, "scaling": "zscore_train_only",
        "random_seed": 3, "n_train": 104, "n_val": 23, "n_test": 23, "num_classes": 3}
TEST_METRICS = {"test_accuracy": 0.9, "test_macro_accuracy": 0.88,
                "test_loss": 0.3, "best_val_loss": 0.25}

REQUIRED_FIELDS = [
    "status", "dataset_path", "split_mode", "evo_random_seed", "data_split_seed",
    "test_accuracy", "test_macro_accuracy", "test_loss", "best_val_loss",
    "best_params", "best_genome", "n_evaluated", "wall_clock_seconds",
    "n_train", "n_val", "n_test", "num_classes", "git_sha", "git_dirty",
    "config_hash", "error",
]


def _ok(**kw):
    base = dict(status=STATUS_OK, run_name="r", config_path="c.yml", cfg=CFG,
                wall_clock_seconds=1.5, meta=META, test_metrics=TEST_METRICS,
                best_genome={"family": "iT"}, best_params=1234.0, n_evaluated=3)
    base.update(kw)
    return build_results(**base)


def test_every_required_field_is_present():
    payload = _ok()
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    assert not missing, f"results.json would omit {missing}"


def test_values_come_from_meta_and_metrics():
    p = _ok()
    assert p["test_accuracy"] == 0.9
    assert p["test_macro_accuracy"] == 0.88
    assert p["n_train"] == 104 and p["n_val"] == 23 and p["n_test"] == 23
    assert p["num_classes"] == 3
    assert p["evo_random_seed"] == 7
    assert p["data_split_seed"] == 3
    assert p["best_params"] == 1234.0
    assert p["n_evaluated"] == 3


def test_split_mode_comes_from_meta_not_config():
    """meta records what the loader did; the config may disagree or be silent."""
    cfg = json.loads(json.dumps(CFG))
    cfg["data"]["tabular"]["split_mode"] = "clean"
    meta = dict(META, split_mode="exaqc", val_is_test=True)
    p = build_results(status=STATUS_OK, run_name="r", config_path="c", cfg=cfg,
                      wall_clock_seconds=1.0, meta=meta, test_metrics=TEST_METRICS)
    assert p["split_mode"] == "exaqc"
    assert p["val_is_test"] is True


def test_data_split_seed_falls_back_to_config_when_meta_lacks_it():
    p = build_results(status=STATUS_OK, run_name="r", config_path="c", cfg=CFG,
                      wall_clock_seconds=1.0, meta={"n_train": 1},
                      test_metrics=TEST_METRICS)
    assert p["data_split_seed"] == 3


def test_failed_record_keeps_identity_and_error():
    p = build_results(status=STATUS_FAILED, run_name="r", config_path="c", cfg=CFG,
                      wall_clock_seconds=0.5, error="RuntimeError: boom")
    assert p["status"] == STATUS_FAILED
    assert p["error"] == "RuntimeError: boom"
    assert p["test_accuracy"] is None
    # Identity must survive so a failure is attributable to a config and seed.
    assert p["evo_random_seed"] == 7
    assert p["dataset_path"] == "data/tabular/iris.csv"
    assert p["config_hash"]


def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "results.json"
    write_results_json(path, _ok())
    assert path.exists()
    leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_written_file_is_strict_json_even_with_nan(tmp_path):
    """NaN macro accuracy must serialise as null, not as the invalid token NaN."""
    p = _ok(test_metrics=dict(TEST_METRICS, test_macro_accuracy=float("nan")))
    path = tmp_path / "results.json"
    write_results_json(path, p)

    raw = path.read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    back = json.loads(raw)          # strict: would raise on NaN
    assert back["test_macro_accuracy"] is None


def test_rewrite_replaces_cleanly(tmp_path):
    path = tmp_path / "results.json"
    write_results_json(path, _ok())
    write_results_json(path, _ok(status=STATUS_FAILED, error="second"))
    assert json.loads(path.read_text())["status"] == STATUS_FAILED


def test_config_hash_is_stable_and_order_independent():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 2, "y": {"a": 3, "b": 2}})


def test_git_info_reports_sha_and_dirty_flag():
    info = git_commit_info(os.getcwd())
    assert set(info) == {"git_sha", "git_dirty"}
    assert info["git_sha"] is None or len(info["git_sha"]) == 40
    assert info["git_dirty"] in (True, False, None)


def test_git_info_outside_a_repo_does_not_raise(tmp_path):
    info = git_commit_info(str(tmp_path))
    assert info["git_sha"] is None
