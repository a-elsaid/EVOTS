"""
Classification metrics must be per-sample, not per-batch.

Every loader uses drop_last=False, so the final batch is usually smaller. Averaging
per-batch rates weights that short batch like a full one. On the real Iris test
split (23 samples, batch_size 16 -> 16 + 7) that reported 0.7723214 where the true
accuracy is 17/23 = 0.7391304: 3.3 points high.

These tests drive the real evaluation entry points with a deterministic stub model,
so they exercise the accumulation in evaluate.py rather than a helper in isolation.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.evaluate import evaluate as ev
from nas_ts.search.genome_init import random_genome

CONFIG = "configs/iris_test.yml"

# 23 samples, batch_size 16 -> batches of 16 and 7.
# Batch 1: 11/16 correct. Batch 2: 6/7 correct.
N_SAMPLES, BATCH_SIZE = 23, 16
N_CORRECT_B1, N_CORRECT_B2 = 11, 6
NUM_CLASSES = 3

TRUE_ACCURACY = (N_CORRECT_B1 + N_CORRECT_B2) / N_SAMPLES          # 17/23 = 0.739130...
BATCH_MEAN_ACCURACY = (N_CORRECT_B1 / BATCH_SIZE +
                       N_CORRECT_B2 / (N_SAMPLES - BATCH_SIZE)) / 2  # 0.772321...


class _FixedPredDataset(Dataset):
    """
    x[0, 0] carries the class the stub model will predict; y is the true class.

    The first BATCH_SIZE samples are the first batch, so correctness is placed
    deliberately: N_CORRECT_B1 right in batch one, N_CORRECT_B2 in batch two.
    """

    def __init__(self):
        self.rows = []
        for i in range(N_SAMPLES):
            in_first = i < BATCH_SIZE
            idx = i if in_first else i - BATCH_SIZE
            want = N_CORRECT_B1 if in_first else N_CORRECT_B2
            true_c = i % NUM_CLASSES
            pred_c = true_c if idx < want else (true_c + 1) % NUM_CLASSES
            self.rows.append((pred_c, true_c))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pred_c, true_c = self.rows[i]
        x = torch.zeros(1, 4, dtype=torch.float32)
        x[0, 0] = float(pred_c)
        return x, torch.tensor(true_c, dtype=torch.long), \
            torch.empty(1, 0), torch.empty(1, 0)


class _StubModel(nn.Module):
    """Predicts the class encoded in x. The parameter is unused, so training
    cannot perturb the predictions and the expected metrics stay exact."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_mark=None, y_mark=None):
        pred = x[:, 0, 0].long().clamp(0, NUM_CLASSES - 1)
        logits = torch.nn.functional.one_hot(pred, NUM_CLASSES).float() * 10.0
        # Zero-weighted so backward() has a graph to walk while the parameter can
        # never move the logits: the expected metrics stay exact across steps.
        return logits + self.unused * 0.0


def _loaders(cfg=None):
    ds = _FixedPredDataset()
    mk = lambda: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    meta = {"d_in": 4, "d_out": 4, "input_length": 1, "num_classes": NUM_CLASSES}
    return mk(), mk(), mk(), meta


@pytest.fixture
def cls_exp_cfg():
    cfg = copy.deepcopy(load_cfg(CONFIG, overrides=None))
    cfg["eval"]["device"] = "cpu"
    cfg["eval"]["training_steps"] = 1
    exp_cfg = build_experiment(cfg)
    exp_cfg.eval_config.datasets[0].loader_fn = _loaders
    exp_cfg.eval_config.datasets[0].loader_kwargs = {}
    return exp_cfg


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    monkeypatch.setattr(ev, "build_model_from_meta", lambda *a, **k: _StubModel())


def test_uneven_batches_would_expose_the_bug():
    """Guard the fixture itself: the two averages must differ clearly."""
    assert N_SAMPLES % BATCH_SIZE != 0
    assert abs(TRUE_ACCURACY - BATCH_MEAN_ACCURACY) > 0.03


def test_search_accuracy_is_per_sample(cls_exp_cfg):
    genome = random_genome(cls_exp_cfg.search_space, cls_exp_cfg.genome_constraints)
    metrics = ev._train_and_eval_on_dataset(genome, cls_exp_cfg)

    assert metrics["accuracy"] == pytest.approx(TRUE_ACCURACY, abs=1e-12)
    assert metrics["accuracy"] != pytest.approx(BATCH_MEAN_ACCURACY, abs=1e-6)
    # correct/total must be a whole number of samples
    assert abs(metrics["accuracy"] * N_SAMPLES - 17) < 1e-9


def test_search_loss_is_per_sample(cls_exp_cfg):
    """Cross-entropy is sample-weighted too: equal per-sample loss here, so the
    mean over an uneven split must equal the per-sample value exactly."""
    genome = random_genome(cls_exp_cfg.search_space, cls_exp_cfg.genome_constraints)
    metrics = ev._train_and_eval_on_dataset(genome, cls_exp_cfg)

    ds = _FixedPredDataset()
    logits, ys = [], []
    for i in range(len(ds)):
        x, y, _, _ = ds[i]
        logits.append(_StubModel()(x.unsqueeze(0)))
        ys.append(y)
    expected = torch.nn.functional.cross_entropy(
        torch.cat(logits), torch.stack(ys), reduction="sum").item() / N_SAMPLES

    # rel=1e-6: float32, and the two sums accumulate in a different order
    # (per batch vs whole split), so they agree only to single precision.
    assert metrics["loss"] == pytest.approx(expected, rel=1e-6)


def test_finetune_test_accuracy_is_per_sample(cls_exp_cfg):
    genome = random_genome(cls_exp_cfg.search_space, cls_exp_cfg.genome_constraints)
    test_metrics, _, _ = ev.finetune_and_test(
        best_fitness=float("inf"),
        genome=genome,
        exp_cfg=cls_exp_cfg,
        initial_state_dict_cpu=None,
        extra_training_steps=2,
    )

    assert test_metrics["test_accuracy"] == pytest.approx(TRUE_ACCURACY, abs=1e-12)
    assert test_metrics["test_accuracy"] != pytest.approx(BATCH_MEAN_ACCURACY, abs=1e-6)
    assert abs(test_metrics["test_accuracy"] * N_SAMPLES - 17) < 1e-9
