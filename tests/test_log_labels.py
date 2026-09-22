"""
Log wording must name the metric that is actually being printed.

finetune_and_test tracks validation in variables called val_mse/best_val_mse for
both task types, but on a classification run the value is cross-entropy. Logging
it as "val_mse" misleads anyone reading a run log. The variables keep their names;
only the logged strings are task-aware.

Also covers the warning when meta lacks 'num_classes', which makes
macro_accuracy unmeasurable (nan).
"""

import copy
import math

import pytest
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.evaluate import evaluate as ev
from nas_ts.search.genome_init import random_genome

CONFIG = "configs/iris_test.yml"
NUM_CLASSES = 3
PRED_LEN, D_OUT, D_IN = 4, 3, 4


@pytest.fixture
def captured():
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="DEBUG")
    yield msgs
    logger.remove(sink)


# ---------------- classification stubs ----------------

class _ClsDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        x = torch.zeros(1, D_IN, dtype=torch.float32)
        x[0, 0] = float(i % NUM_CLASSES)
        return x, torch.tensor(i % NUM_CLASSES, dtype=torch.long), \
            torch.empty(1, 0), torch.empty(1, 0)


class _ClsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_mark=None, y_mark=None):
        pred = x[:, 0, 0].long().clamp(0, NUM_CLASSES - 1)
        logits = torch.nn.functional.one_hot(pred, NUM_CLASSES).float() * 10.0
        return logits + self.unused * 0.0


# ---------------- forecasting stubs ----------------

class _FcDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        x = torch.full((8, D_IN), float(i), dtype=torch.float32)
        y = torch.full((PRED_LEN, D_OUT), float(i), dtype=torch.float32)
        return x, y, torch.empty(8, 0), torch.empty(PRED_LEN, 0)


class _FcModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, x_mark=None, y_mark=None):
        b = x.size(0)
        return torch.zeros(b, PRED_LEN, D_OUT) + self.scale * 0.0


def _exp_cfg(task_type, meta):
    is_cls = task_type == "classification"
    ds = _ClsDataset() if is_cls else _FcDataset()

    def loaders():
        mk = lambda: DataLoader(ds, batch_size=8, shuffle=False, drop_last=False)
        return mk(), mk(), mk(), meta

    cfg = copy.deepcopy(load_cfg(CONFIG, overrides=None))
    cfg["eval"]["device"] = "cpu"
    cfg["eval"]["training_steps"] = 1
    if not is_cls:
        cfg["task"]["task_type"] = "forecasting"
        cfg["task"]["pred_length"] = PRED_LEN
        cfg["task"]["d_out"] = D_OUT
        cfg["data"] = {"csv": {"paths": ["unused.csv"], "train_ratio": 0.7,
                               "val_ratio": 0.1, "batch_size": 8,
                               "num_workers": 0, "normalize": True}}
        cfg["selection"]["objectives"] = ["mse"]
    exp_cfg = build_experiment(cfg)
    exp_cfg.eval_config.datasets[0].loader_fn = loaders
    exp_cfg.eval_config.datasets[0].loader_kwargs = {}
    return exp_cfg


def _run_finetune(exp_cfg, model_cls):
    import nas_ts.evaluate.evaluate as mod
    orig = mod.build_model_from_meta
    mod.build_model_from_meta = lambda *a, **k: model_cls()
    try:
        genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
        return mod.finetune_and_test(
            best_fitness=float("inf"), genome=genome, exp_cfg=exp_cfg,
            initial_state_dict_cpu=None, extra_training_steps=3,
        )
    finally:
        mod.build_model_from_meta = orig


CLS_META = {"d_in": D_IN, "d_out": D_OUT, "input_length": 1, "num_classes": NUM_CLASSES}
FC_META = {"d_in": D_IN, "d_out": D_OUT, "input_length": 8}


def test_classification_finetune_logs_val_loss_not_val_mse(captured):
    _run_finetune(_exp_cfg("classification", CLS_META), _ClsModel)
    text = "\n".join(captured)

    assert "val_loss" in text, "classification run never logged val_loss"
    finetune_lines = [m for m in captured if "[Finetune]" in m or "[EarlyStop]" in m]
    offenders = [m.strip() for m in finetune_lines if "val_mse" in m]
    assert not offenders, f"classification run still logs val_mse: {offenders}"


def test_forecasting_finetune_still_logs_val_mse(captured):
    _run_finetune(_exp_cfg("forecasting", FC_META), _FcModel)
    text = "\n".join(captured)

    assert "val_mse" in text, "forecasting wording changed; it must stay val_mse"
    assert "val_loss" not in text


def test_missing_num_classes_warns_and_names_the_key(captured):
    meta_without = {k: v for k, v in CLS_META.items() if k != "num_classes"}
    test_metrics, _, _ = _run_finetune(_exp_cfg("classification", meta_without), _ClsModel)

    warnings = [m for m in captured if "num_classes" in m and "WARNING" in m]
    assert warnings, "no warning raised when meta lacked num_classes"
    assert "macro_accuracy" in warnings[0]
    assert "nan" in warnings[0], "warning must say the value is reported as nan"
    # nan, not 0.0: an unmeasurable metric must not read as a plausible score.
    assert math.isnan(test_metrics["test_macro_accuracy"])


def test_present_num_classes_does_not_warn(captured):
    _run_finetune(_exp_cfg("classification", CLS_META), _ClsModel)
    assert not [m for m in captured if "missing 'num_classes'" in m]


def test_forecasting_never_warns_about_num_classes(captured):
    _run_finetune(_exp_cfg("forecasting", FC_META), _FcModel)
    assert not [m for m in captured if "num_classes" in m]
