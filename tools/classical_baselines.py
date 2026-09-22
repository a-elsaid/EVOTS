#!/usr/bin/env python3
"""
Classical baselines on exactly the splits EvoTS is evaluated on.

Without these, "EvoTS beats EXAQC" is unanchored: on datasets this size a
logistic regression is often within noise of anything, and a reader's first
question is what the cheap model gets. The splits are built by calling
make_tabular_dataloaders through the very config EvoTS uses, so the samples are
identical by construction rather than by a reimplementation that agrees today.

    python tools/classical_baselines.py
    python tools/classical_baselines.py --datasets iris --seeds 0 1 --models logreg

Hyperparameters are chosen on validation, never on test. In "exaqc" mode
validation IS test, so no selection happens at all there and library defaults
are used -- see HYPERPARAMETER NOTE below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nas_ts.core.config_loader import build_experiment, load_cfg  # noqa: E402
from nas_ts.evaluate.evaluate import macro_accuracy  # noqa: E402
from nas_ts.utils.run_results import (  # noqa: E402
    RESULTS_FILENAME, STATUS_FAILED, STATUS_OK, config_hash, git_commit_info,
    write_results_json,
)

ALL_DATASETS = ["iris", "wine", "seeds", "breast_cancer"]
ALL_SPLIT_MODES = ["clean", "exaqc"]
ALL_MODELS = ["logreg", "svc_rbf", "mlp"]
DEFAULT_SEEDS = list(range(10))

CONFIG_DIR = REPO_ROOT / "configs" / "classification"

# HYPERPARAMETER NOTE
# Small grids, selected on validation accuracy. In "exaqc" mode the validation
# split IS the test split, so selecting on it would be selecting on test; there
# the first (default) candidate is used with no selection. That makes the exaqc
# baselines slightly conservative relative to EvoTS in the same mode, which does
# select on that holdout. The JSON records which happened, so the comparison
# table can say so.
GRIDS = {
    "logreg": [{"C": c} for c in (1.0, 0.01, 0.1, 10.0, 100.0)],
    "svc_rbf": [{"C": c, "gamma": g}
                for c in (1.0, 0.1, 10.0, 100.0) for g in ("scale", 0.01, 0.1)],
    "mlp": [{"hidden_layer_sizes": h, "alpha": a}
            for h in ((64,), (32,), (64, 32)) for a in (1e-4, 1e-2)],
}


def build_model(name: str, params: dict, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=5000, random_state=seed, **params)
    if name == "svc_rbf":
        # No probability=True: it is deprecated in scikit-learn 1.9 and removed
        # in 1.11, so asking for it would make this script's behaviour depend on
        # the cluster's version. SVC therefore reports no log loss (see
        # _predict_proba_or_none); accuracy and macro accuracy, which are what
        # the EXAQC comparison uses, are unaffected.
        return SVC(kernel="rbf", random_state=seed, **params)
    if name == "mlp":
        return MLPClassifier(max_iter=2000, random_state=seed, **params)
    raise ValueError(f"Unknown model {name!r}. Known: {', '.join(ALL_MODELS)}")


def count_parameters(name: str, model) -> float:
    """Fitted parameter count, for a rough size comparison against the NAS models."""
    try:
        if name == "logreg":
            return float(model.coef_.size + model.intercept_.size)
        if name == "svc_rbf":
            return float(model.dual_coef_.size + model.intercept_.size
                         + model.support_vectors_.size)
        if name == "mlp":
            return float(sum(c.size for c in model.coefs_)
                         + sum(b.size for b in model.intercepts_))
    except Exception:
        pass
    return None


def load_splits(dataset: str, split_mode: str, seed: int):
    """
    The splits EvoTS sees, produced by EvoTS's own loader.

    Going through load_cfg/build_experiment means the ratios, scaler, stratified
    split and seed handling are the same code, not a copy that can drift.
    """
    cfg = load_cfg(str(CONFIG_DIR / f"{dataset}.yml"), overrides=[
        f"data.tabular.random_seed={seed}",
        f"data.tabular.split_mode={split_mode}",
    ])
    exp_cfg = build_experiment(cfg)
    ds_cfg = exp_cfg.eval_config.datasets[0]
    train_loader, val_loader, test_loader, meta = ds_cfg.loader_fn(**ds_cfg.loader_kwargs)

    def flatten(loader):
        """
        Samples in dataset order.

        Read from the dataset, not by iterating the loader: the train loader has
        shuffle=True, so iterating it returns the same samples in a different
        order every call, and that order changes lbfgs and MLP numerics enough
        to make the same seed disagree with itself between invocations. Sklearn
        does not need batches, so the shuffling serves nothing here.
        """
        ds = loader.dataset
        if hasattr(ds, "X") and hasattr(ds, "y"):
            return ds.X.numpy(), ds.y.numpy()

        # Fallback for any other dataset type: one pass, X and y together, so a
        # shuffled loader cannot pair features with another permutation's labels.
        xs, ys = [], []
        for batch in loader:
            xs.append(batch[0].squeeze(1).numpy())   # [B, 1, F] -> [B, F]
            ys.append(batch[1].numpy())
        return np.concatenate(xs), np.concatenate(ys)

    return flatten(train_loader), flatten(val_loader), flatten(test_loader), meta, cfg


def _predict_proba_or_none(model, X):
    """Probabilities when the estimator provides them, else None (SVC)."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def evaluate(model, X, y, num_classes: int):
    """
    Accuracy, macro accuracy and log loss, using EvoTS's own macro_accuracy so
    the definition cannot drift from the numbers the NAS runs report.

    Loss is None for estimators without calibrated probabilities.
    """
    preds = model.predict(X)
    correct = torch.zeros(num_classes, dtype=torch.long)
    total = torch.zeros(num_classes, dtype=torch.long)
    for true, pred in zip(y, preds):
        total[int(true)] += 1
        correct[int(true)] += int(true == pred)

    accuracy = float((preds == y).mean())
    proba = _predict_proba_or_none(model, X)
    loss = (float(log_loss(y, proba, labels=list(range(num_classes))))
            if proba is not None else None)
    return accuracy, macro_accuracy(correct, total), loss


def run_one(model_name: str, dataset: str, split_mode: str, seed: int,
            out_dir: Path) -> dict:
    started = time.perf_counter()
    (train_X, train_y), (val_X, val_y), (test_X, test_y), meta, cfg = \
        load_splits(dataset, split_mode, seed)
    num_classes = int(meta["num_classes"])

    grid = GRIDS[model_name]
    tuned_on_val = (split_mode != "exaqc")
    candidates = grid if tuned_on_val else grid[:1]

    best = None
    for params in candidates:
        model = build_model(model_name, params, seed)
        model.fit(train_X, train_y)
        # Selection is on accuracy, which every estimator provides; loss is
        # recorded but not selected on, so a None loss changes nothing.
        val_acc, _, val_loss = evaluate(model, val_X, val_y, num_classes)
        if best is None or val_acc > best["val_acc"]:
            best = {"params": params, "model": model,
                    "val_acc": val_acc, "val_loss": val_loss}

    # Test is touched exactly once, with the already-chosen model.
    test_acc, test_macro, test_loss = evaluate(best["model"], test_X, test_y, num_classes)

    payload = {
        "status": STATUS_OK,
        "run_name": run_name(model_name, dataset, split_mode, seed),
        "model": model_name,
        "model_family": "classical_baseline",
        "config_path": str(CONFIG_DIR / f"{dataset}.yml"),
        "task_type": "classification",

        "dataset_path": cfg["data"]["tabular"]["path"],
        "split_mode": meta.get("split_mode"),
        "val_is_test": meta.get("val_is_test"),
        "scaling": meta.get("scaling"),

        "evo_random_seed": None,          # no evolutionary search here
        "data_split_seed": meta.get("random_seed", seed),

        "test_accuracy": test_acc,
        "test_macro_accuracy": test_macro,
        "test_loss": test_loss,
        "best_val_loss": best["val_loss"],

        "best_params": count_parameters(model_name, best["model"]),
        "best_genome": None,
        "n_evaluated": len(candidates),
        "wall_clock_seconds": round(time.perf_counter() - started, 3),

        "n_train": meta.get("n_train"),
        "n_val": meta.get("n_val"),
        "n_test": meta.get("n_test"),
        "num_classes": num_classes,

        "hyperparameters": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in best["params"].items()},
        "hyperparameter_selection": (
            "validation accuracy" if tuned_on_val
            else "none (exaqc mode: validation is test, so selecting there would select on test)"
        ),
        "val_accuracy": best["val_acc"],

        "config_hash": config_hash(cfg),
        "error": None,
    }
    payload.update(git_commit_info(str(REPO_ROOT)))
    return payload


def run_name(model: str, dataset: str, split_mode: str, seed: int) -> str:
    return f"baseline-{model}_{dataset}_{split_mode}_seed{seed}"


def results_file(out_dir: Path, name: str) -> Path:
    return out_dir / name / RESULTS_FILENAME


def already_ok(path: Path) -> bool:
    try:
        return json.loads(path.read_text()).get("status") == STATUS_OK
    except Exception:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Classical baselines on the same splits as the EvoTS runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--split-modes", nargs="+", default=ALL_SPLIT_MODES,
                   choices=ALL_SPLIT_MODES)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--out-dir", default="logs/classification")
    p.add_argument("--force", action="store_true",
                   help="recompute even if results.json already says ok")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    plan = [(m, d, s, seed)
            for m in args.models
            for d in args.datasets
            for s in args.split_modes
            for seed in args.seeds]

    print(f"[Baselines] {len(plan)} runs: {len(args.models)} models x "
          f"{len(args.datasets)} datasets x {len(args.split_modes)} split modes x "
          f"{len(args.seeds)} seeds")
    print(f"[Baselines] Output: {out_dir.resolve()}")

    if args.dry_run:
        for m, d, s, seed in plan:
            name = run_name(m, d, s, seed)
            state = "skip (ok)" if (not args.force and already_ok(results_file(out_dir, name))) else "run"
            print(f"    {name:52s} {state}")
        return 0

    ok, skipped, failed = 0, 0, []
    for i, (model_name, dataset, split_mode, seed) in enumerate(plan, start=1):
        name = run_name(model_name, dataset, split_mode, seed)
        rpath = results_file(out_dir, name)

        if not args.force and already_ok(rpath):
            skipped += 1
            continue

        try:
            payload = run_one(model_name, dataset, split_mode, seed, out_dir)
            write_results_json(rpath, payload)
            ok += 1
            print(f"[{i}/{len(plan)}] {name}: acc={payload['test_accuracy']:.4f} "
                  f"macro={payload['test_macro_accuracy']:.4f}")
        except Exception as e:
            # Same contract as the NAS suite: a failure is recorded, never silent.
            failed.append(name)
            write_results_json(rpath, {
                "status": STATUS_FAILED,
                "run_name": name,
                "model": model_name,
                "model_family": "classical_baseline",
                "dataset_path": f"data/tabular/{dataset}.csv",
                "split_mode": split_mode,
                "data_split_seed": seed,
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"[{i}/{len(plan)}] {name}: FAILED {type(e).__name__}: {e}")

    print(f"\n[Baselines] {ok} ok | {len(failed)} failed | {skipped} skipped "
          f"| {len(plan)} planned")
    for name in failed:
        print(f"    failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
