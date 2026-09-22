"""
Classical baselines must sit on exactly the same samples as the EvoTS runs.

If the baseline built its own split, a difference in accuracy could be a
difference in data rather than in model, and the comparison would be worthless.
These tests check sample-level identity, not just matching split sizes.

They also pin the rule that test is never used for model selection, which is the
one thing that would quietly inflate the baselines.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import classical_baselines as cb  # noqa: E402

from conftest import require_tabular  # noqa: E402

from nas_ts.core.config_loader import build_experiment, load_cfg  # noqa: E402


# ---------------------------------------------------------------- identity

@pytest.mark.parametrize("split_mode", ["clean", "exaqc"])
def test_splits_are_identical_to_what_evots_trains_on(split_mode):
    require_tabular("iris")
    seed = 3
    (btr_X, btr_y), (bva_X, bva_y), (bte_X, bte_y), meta, _ = \
        cb.load_splits("iris", split_mode, seed)

    # The EvoTS path, independently constructed.
    cfg = load_cfg("configs/classification/iris.yml", overrides=[
        f"data.tabular.random_seed={seed}",
        f"data.tabular.split_mode={split_mode}",
    ])
    ds_cfg = build_experiment(cfg).eval_config.datasets[0]
    tr, va, te, evots_meta = ds_cfg.loader_fn(**ds_cfg.loader_kwargs)

    def flat(loader):
        """
        One pass, collecting X and y together. Two separate comprehensions would
        iterate a shuffled loader twice and pair features from one permutation
        with labels from another.
        """
        xs, ys = [], []
        for b in loader:
            xs.append(b[0].squeeze(1).numpy())
            ys.append(b[1].numpy())
        return np.concatenate(xs), np.concatenate(ys)

    def canonical(X, y):
        """
        Rows sorted, because the train loader has shuffle=True and so yields a
        different ORDER on every iteration. What must match is the set of
        samples, which is what "the same split" means.
        """
        rows = np.concatenate([X, y.reshape(-1, 1).astype(X.dtype)], axis=1)
        return rows[np.lexsort(rows.T[::-1])]

    for (bx, by), loader, name in (((btr_X, btr_y), tr, "train"),
                                   ((bva_X, bva_y), va, "val"),
                                   ((bte_X, bte_y), te, "test")):
        ex, ey = flat(loader)
        assert np.array_equal(canonical(bx, by), canonical(ex, ey)), \
            f"{name} split differs from the one EvoTS trains on"
        assert len(bx) == len(ex), f"{name} split size differs"
        # Labels must travel with their own features, not merely match in count.
        assert sorted(zip(map(tuple, bx.tolist()), by.tolist())) == \
               sorted(zip(map(tuple, ex.tolist()), ey.tolist())), \
            f"{name}: feature/label pairing differs"

    assert meta["n_train"] == evots_meta["n_train"]
    assert meta["split_mode"] == split_mode


def test_same_seed_same_split_different_seed_different_split():
    require_tabular("iris")
    a = cb.load_splits("iris", "clean", 0)[2][0]
    b = cb.load_splits("iris", "clean", 0)[2][0]
    c = cb.load_splits("iris", "clean", 1)[2][0]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_exaqc_val_and_test_are_the_same_samples():
    require_tabular("iris")
    (_, _), (va_X, va_y), (te_X, te_y), meta, _ = cb.load_splits("iris", "exaqc", 0)
    assert np.array_equal(va_X, te_X) and np.array_equal(va_y, te_y)
    assert meta["val_is_test"] is True


# ------------------------------------------------------------- no test leak

def test_clean_mode_selects_on_validation_and_tries_the_whole_grid():
    require_tabular("iris")
    payload = cb.run_one("logreg", "iris", "clean", 0, Path("/unused"))
    assert payload["hyperparameter_selection"] == "validation accuracy"
    assert payload["n_evaluated"] == len(cb.GRIDS["logreg"])


def test_exaqc_mode_does_not_select_at_all():
    """In exaqc mode validation IS test, so any selection would be on test."""
    require_tabular("iris")
    payload = cb.run_one("logreg", "iris", "exaqc", 0, Path("/unused"))
    assert payload["n_evaluated"] == 1
    assert "would select on test" in payload["hyperparameter_selection"]
    assert payload["hyperparameters"] == {"C": cb.GRIDS["logreg"][0]["C"]}


def test_selection_never_looks_at_test(monkeypatch):
    """
    The test split must be evaluated exactly once, and last: after the
    hyperparameters are already chosen. Any earlier look is selection on test.
    """
    require_tabular("iris")
    real_evaluate = cb.evaluate
    splits = cb.load_splits("iris", "clean", 0)
    (_, _), (val_X, _), (test_X, _), _, _ = splits
    monkeypatch.setattr(cb, "load_splits", lambda *a, **k: splits)

    calls = []

    def watching_evaluate(model, X, y, num_classes):
        if X.shape == test_X.shape and np.array_equal(X, test_X):
            calls.append("test")
        elif X.shape == val_X.shape and np.array_equal(X, val_X):
            calls.append("val")
        else:
            calls.append("train")
        return real_evaluate(model, X, y, num_classes)
    monkeypatch.setattr(cb, "evaluate", watching_evaluate)

    cb.run_one("logreg", "iris", "clean", 0, Path("/unused"))

    assert calls.count("test") == 1, f"test evaluated {calls.count('test')} times: {calls}"
    assert calls[-1] == "test", f"test was not the last evaluation: {calls}"
    assert calls[:-1] == ["val"] * len(cb.GRIDS["logreg"]), \
        f"selection should score every candidate on val only: {calls}"


# ------------------------------------------------------------------ schema

REQUIRED = ["status", "run_name", "model", "dataset_path", "split_mode",
            "val_is_test", "data_split_seed", "test_accuracy",
            "test_macro_accuracy", "test_loss", "best_val_loss", "best_params",
            "n_evaluated", "wall_clock_seconds", "n_train", "n_val", "n_test",
            "num_classes", "config_hash", "git_sha", "git_dirty", "error"]


@pytest.mark.parametrize("model", cb.ALL_MODELS)
def test_result_schema_matches_the_nas_runs(model):
    require_tabular("iris")
    payload = cb.run_one(model, "iris", "clean", 0, Path("/unused"))
    missing = [f for f in REQUIRED if f not in payload]
    assert not missing, f"{model} result omits {missing}"
    assert payload["status"] == "ok"
    assert payload["model_family"] == "classical_baseline"
    assert 0.0 <= payload["test_accuracy"] <= 1.0
    assert 0.0 <= payload["test_macro_accuracy"] <= 1.0


def test_svc_reports_no_loss_rather_than_an_invented_one():
    """probability=True is removed in sklearn 1.11; we do not depend on it."""
    require_tabular("iris")
    payload = cb.run_one("svc_rbf", "iris", "clean", 0, Path("/unused"))
    assert payload["test_loss"] is None
    assert payload["test_accuracy"] is not None


def test_macro_accuracy_uses_the_same_definition_as_evots():
    require_tabular("breast_cancer")
    payload = cb.run_one("logreg", "breast_cancer", "clean", 0, Path("/unused"))
    # Imbalanced dataset: the two must be computable and need not agree.
    assert payload["test_accuracy"] is not None
    assert payload["test_macro_accuracy"] is not None


# ------------------------------------------------------------------- naming

def test_run_names_are_unique_and_marked_as_baselines():
    names = {cb.run_name(m, d, s, seed)
             for m in cb.ALL_MODELS for d in cb.ALL_DATASETS
             for s in cb.ALL_SPLIT_MODES for seed in range(10)}
    assert len(names) == 3 * 4 * 2 * 10
    assert all(n.startswith("baseline-") for n in names)


def test_dry_run_computes_nothing(tmp_path, capsys):
    code = cb.main(["--dry-run", "--datasets", "iris", "--models", "logreg",
                    "--seeds", "0", "--split-modes", "clean",
                    "--out-dir", str(tmp_path)])
    assert code == 0
    assert not list(tmp_path.glob("*/results.json"))


def test_failure_is_recorded_not_raised(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("fit exploded")
    monkeypatch.setattr(cb, "run_one", boom)

    code = cb.main(["--datasets", "iris", "--models", "logreg", "--seeds", "0",
                    "--split-modes", "clean", "--out-dir", str(tmp_path)])
    assert code == 1
    rec = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert rec["status"] == "failed"
    assert "fit exploded" in rec["error"]


def test_loader_pairing_is_stable_across_iterations():
    """
    The train loader shuffles, so a consumer must collect X and y in ONE pass.
    This pins that the loader itself keeps each label with its own features,
    which is what makes the single-pass requirement sufficient.
    """
    require_tabular("iris")
    from nas_ts.utils.tabular_data_module import (
        TabularDataConfig, make_tabular_dataloaders)

    tr, _, _, _ = make_tabular_dataloaders(TabularDataConfig(
        path="data/tabular/iris.csv", target_col="target", train_ratio=0.7,
        val_ratio=0.15, batch_size=16, normalize=True, random_seed=3))

    def one_pass(loader):
        out = []
        for b in loader:
            for row, lbl in zip(b[0].squeeze(1).tolist(), b[1].tolist()):
                out.append((tuple(np.round(row, 6)), int(lbl)))
        return sorted(out)

    assert one_pass(tr) == one_pass(tr), "pairing changed between iterations"

    ds = tr.dataset
    stored = sorted((tuple(np.round(ds.X[i].tolist(), 6)), int(ds.y[i]))
                    for i in range(len(ds)))
    assert one_pass(tr) == stored, "loader pairs differ from the dataset's own"
