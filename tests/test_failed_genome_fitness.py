"""
A genome that raises during evaluation must be scored inf, not abort the search.

The backends return a hardcoded failure metrics dict when a worker raises. That
dict used to be forecasting-only ({"mse", "mae", "params"}), so on a
classification run the fitness aggregator's metrics["loss"] raised KeyError and
one bad genome killed the whole search.

These tests drive the real failure path -- the backends' own worker/poll code and
the real fitness aggregator -- rather than the failure-dict helper alone.
"""

import copy
import time

import pytest

from nas_ts.backends import backend_process, backend_threaded
from nas_ts.backends.backend_threaded import ThreadedBackend
from nas_ts.core.config_loader import (
    aggregate_classification_fitness,
    build_experiment,
    load_cfg,
)
from nas_ts.search.engine import EvolutionEngine
from nas_ts.search.genome_init import random_genome

CONFIG = "configs/iris_test.yml"


class Boom(RuntimeError):
    pass


def _exp_cfg(task_type="classification"):
    cfg = copy.deepcopy(load_cfg(CONFIG, overrides=None))
    cfg["eval"]["device"] = "cpu"
    if task_type == "forecasting":
        # Flip the task type and give it the data section build_experiment wants.
        # The evaluator is monkeypatched to raise, so nothing is ever loaded from
        # this path -- only the task type decides which failure keys come back.
        cfg["task"]["task_type"] = "forecasting"
        cfg["data"] = {"csv": {
            "paths": ["unused.csv"],
            "train_ratio": 0.7, "val_ratio": 0.1,
            "batch_size": 16, "num_workers": 0, "normalize": True,
        }}
        cfg["selection"]["objectives"] = ["mse"]
    return build_experiment(cfg)


def _a_genome(exp_cfg):
    return random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)


def _drain(backend, expected, timeout=30.0):
    """Poll until `expected` results arrive; the backend is non-blocking."""
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        out.extend(backend.poll())
        if len(out) >= expected:
            break
        time.sleep(0.01)
    return out


def test_threaded_backend_failure_scores_inf_for_classification(monkeypatch):
    exp_cfg = _exp_cfg("classification")
    monkeypatch.setattr(
        backend_threaded, "evaluate_genome",
        lambda *a, **k: (_ for _ in ()).throw(Boom("genome blew up")),
    )

    backend = ThreadedBackend(exp_cfg=exp_cfg, max_workers=1)
    try:
        backend.submit("bad-0", _a_genome(exp_cfg))
        results = _drain(backend, 1)
    finally:
        backend.shutdown(wait=True)

    assert len(results) == 1
    _, metrics = results[0]

    # The key the classification aggregator actually reads.
    assert "loss" in metrics, f"failure dict lacks 'loss': {sorted(metrics)}"
    assert metrics["loss"] == float("inf")
    assert "worker_error" in metrics

    # The real aggregator, on the real failure dict: inf, no KeyError.
    assert aggregate_classification_fitness(metrics) == float("inf")


def test_threaded_backend_failure_keeps_forecasting_keys(monkeypatch):
    exp_cfg = _exp_cfg("forecasting")
    monkeypatch.setattr(
        backend_threaded, "evaluate_genome",
        lambda *a, **k: (_ for _ in ()).throw(Boom("genome blew up")),
    )

    backend = ThreadedBackend(exp_cfg=exp_cfg, max_workers=1)
    try:
        backend.submit("bad-0", _a_genome(exp_cfg))
        results = _drain(backend, 1)
    finally:
        backend.shutdown(wait=True)

    _, metrics = results[0]
    # Byte-for-byte the pre-existing forecasting failure dict.
    assert metrics["mse"] == float("inf")
    assert metrics["mae"] == float("inf")
    assert metrics["params"] == float("inf")
    assert "loss" not in metrics


def test_process_worker_failure_scores_inf_for_classification(monkeypatch):
    """The function the process pool actually runs, called in-process."""
    exp_cfg = _exp_cfg("classification")
    monkeypatch.setattr(backend_process, "_G_EXP_CFG", exp_cfg)
    monkeypatch.setattr(
        backend_process, "evaluate_genome",
        lambda *a, **k: (_ for _ in ()).throw(Boom("genome blew up")),
    )

    indiv_id, metrics = backend_process._process_worker("bad-0", _a_genome(exp_cfg))

    assert indiv_id == "bad-0"
    assert metrics["loss"] == float("inf")
    assert "mse" not in metrics
    assert aggregate_classification_fitness(metrics) == float("inf")


def test_search_survives_every_genome_failing(monkeypatch, tmp_path):
    """
    End to end: real engine, real backend, every evaluation raising.

    Before the fix this died with KeyError: 'loss' on the first completion.
    """
    exp_cfg = _exp_cfg("classification")
    monkeypatch.setattr(
        backend_threaded, "evaluate_genome",
        lambda *a, **k: (_ for _ in ()).throw(Boom("genome blew up")),
    )

    backend = ThreadedBackend(exp_cfg=exp_cfg, max_workers=1)
    engine = EvolutionEngine(
        exp_cfg=exp_cfg,
        backend=backend,
        logger_=None,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    try:
        final_pop, best = engine.run()
    finally:
        backend.shutdown(wait=True)

    evaluated = [i for i in final_pop.get_all() if i.fitness is not None]
    assert evaluated, "search produced no scored individuals"
    assert all(i.fitness == float("inf") for i in evaluated)
    assert best is not None
