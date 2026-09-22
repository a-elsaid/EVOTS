"""
Macro (mean-class) accuracy, to match EXAQC's reported metric.

Plain accuracy is micro: correct/total over the split, so a large class dominates.
Macro takes accuracy per class and averages over classes, so every class counts
equally. On an imbalanced split the two differ, and EXAQC reports the macro one --
comparing our micro against their macro would not be like-for-like.

The tests drive the real evaluation entry points with a deterministic stub model.
"""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.evaluate import evaluate as ev
from nas_ts.search.genome_init import random_genome

CONFIG = "configs/iris_test.yml"
NUM_CLASSES = 3
BATCH_SIZE = 16

# (class, n_samples, n_correct) -- deliberately imbalanced, and the split size is
# not a multiple of BATCH_SIZE so this also rides on the Task 2 per-sample fix.
IMBALANCED = [(0, 10, 9), (1, 8, 4), (2, 2, 0)]
MICRO = (9 + 4 + 0) / (10 + 8 + 2)                      # 13/20 = 0.65
MACRO = (9 / 10 + 4 / 8 + 0 / 2) / 3                    # 1.4/3  = 0.466666...

# One class entirely absent from the split.
ABSENT = [(0, 10, 9), (1, 6, 3)]
ABSENT_MICRO = (9 + 3) / (10 + 6)                       # 12/16 = 0.75
ABSENT_MACRO = (9 / 10 + 3 / 6) / 2                     # 0.70  (averaged over 2)


class _SpecDataset(Dataset):
    """x[0, 0] carries the class the stub model predicts; y is the true class."""

    def __init__(self, spec):
        self.rows = []
        for true_c, n, n_correct in spec:
            for i in range(n):
                pred_c = true_c if i < n_correct else (true_c + 1) % NUM_CLASSES
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
    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_mark=None, y_mark=None):
        pred = x[:, 0, 0].long().clamp(0, NUM_CLASSES - 1)
        logits = torch.nn.functional.one_hot(pred, NUM_CLASSES).float() * 10.0
        return logits + self.unused * 0.0


def _exp_cfg_for(spec):
    def loaders():
        ds = _SpecDataset(spec)
        mk = lambda: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        meta = {"d_in": 4, "d_out": 4, "input_length": 1, "num_classes": NUM_CLASSES}
        return mk(), mk(), mk(), meta

    cfg = copy.deepcopy(load_cfg(CONFIG, overrides=None))
    cfg["eval"]["device"] = "cpu"
    cfg["eval"]["training_steps"] = 1
    exp_cfg = build_experiment(cfg)
    exp_cfg.eval_config.datasets[0].loader_fn = loaders
    exp_cfg.eval_config.datasets[0].loader_kwargs = {}
    return exp_cfg


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    monkeypatch.setattr(ev, "build_model_from_meta", lambda *a, **k: _StubModel())


def test_micro_and_macro_differ_in_the_fixture():
    """Guard the fixture: a split where the two agree would prove nothing."""
    assert abs(MICRO - MACRO) > 0.15


def test_search_metrics_report_both_accuracies():
    exp_cfg = _exp_cfg_for(IMBALANCED)
    genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
    metrics = ev._train_and_eval_on_dataset(genome, exp_cfg)

    assert metrics["accuracy"] == pytest.approx(MICRO, abs=1e-12)
    assert metrics["macro_accuracy"] == pytest.approx(MACRO, abs=1e-12)
    assert metrics["accuracy"] != pytest.approx(metrics["macro_accuracy"], abs=1e-6)


def test_finetune_metrics_report_both_accuracies():
    exp_cfg = _exp_cfg_for(IMBALANCED)
    genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
    test_metrics, _, _ = ev.finetune_and_test(
        best_fitness=float("inf"),
        genome=genome,
        exp_cfg=exp_cfg,
        initial_state_dict_cpu=None,
        extra_training_steps=2,
    )

    assert test_metrics["test_accuracy"] == pytest.approx(MICRO, abs=1e-12)
    assert test_metrics["test_macro_accuracy"] == pytest.approx(MACRO, abs=1e-12)


def test_absent_class_is_skipped_not_scored_zero():
    """
    A class with no sample in the split is left out of the average. Scoring it 0
    would report 0.4666 here instead of 0.70, penalising the model for the split.
    """
    exp_cfg = _exp_cfg_for(ABSENT)
    genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
    metrics = ev._train_and_eval_on_dataset(genome, exp_cfg)

    assert metrics["accuracy"] == pytest.approx(ABSENT_MICRO, abs=1e-12)
    assert metrics["macro_accuracy"] == pytest.approx(ABSENT_MACRO, abs=1e-12)
    # not the "count absent as zero" value
    assert metrics["macro_accuracy"] != pytest.approx((0.9 + 0.5 + 0.0) / 3, abs=1e-6)


def test_macro_accuracy_does_not_feed_fitness():
    """Fitness stays cross-entropy loss; macro is reported only."""
    from nas_ts.core.config_loader import aggregate_classification_fitness

    exp_cfg = _exp_cfg_for(IMBALANCED)
    genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
    metrics = ev._train_and_eval_on_dataset(genome, exp_cfg)

    assert aggregate_classification_fitness(metrics) == metrics["loss"]
    assert aggregate_classification_fitness(metrics) != metrics["macro_accuracy"]
