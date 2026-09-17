"""Regression tests for finetune_and_test() early stopping and weight restoration.

Two defects, both of which corrupted the final reported number of a run:

  1. The "validation did not improve" case was split across two branches, and the
     branch that fired when val_mse was worse than the search best did not
     decrement patience. Since the other branch required exact float equality,
     patience never reached zero and early stopping could never trigger — every
     run burned all extra_training_steps.

  2. When finetune never improved, the else branch reassigned best_state_cpu but
     never loaded those weights back into `model`. Test evaluation therefore ran
     on the drifted final-step weights while the state_dict that was returned
     (and saved as {run}__best_finetuned.pt) held different weights, so test_mse
     and the saved checkpoint described two different models.

Everything here runs on CPU with random tensors in a couple of seconds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nas_ts.core.config import DatasetConfig, EvalConfig, TaskConfig
from nas_ts.core.genome_v2 import BlockSpec, Genome, StageSpec
from nas_ts.evaluate.evaluate import finetune_and_test
from nas_ts.models.model_builder_v2 import build_model_from_meta


MODEL_DIM = 16
D_IN = D_OUT = 3
INPUT_LENGTH = 16
PRED_LENGTH = 4
BATCH_SIZE = 2
N_BATCHES = 3
MARK_DIM = 4

MAX_STEPS = 40
PATIENCE = 2


def _task() -> TaskConfig:
    return TaskConfig(
        task_type="forecasting",
        input_length=INPUT_LENGTH,
        pred_length=PRED_LENGTH,
        d_in=D_IN,
        d_out=D_OUT,
        metrics=["mse"],
        use_norm=False,
    )


def _genome() -> Genome:
    return Genome(
        family="test",
        model_dim=MODEL_DIM,
        num_heads=2,
        ff_mult=2.0,
        pos_encoding="none",
        dropout=0.0,
        stages=[StageSpec(name="stage0", tokenizer="time",
                          blocks=[BlockSpec(block_type="attn")])],
    )


def _batches(n: int, seed: int):
    """4-tuples: finetune_and_test calls .to(device) on the marks unconditionally."""
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(BATCH_SIZE, INPUT_LENGTH, D_IN, generator=g),
            torch.randn(BATCH_SIZE, PRED_LENGTH, D_OUT, generator=g),
            torch.randn(BATCH_SIZE, INPUT_LENGTH, MARK_DIM, generator=g),
            torch.randn(BATCH_SIZE, PRED_LENGTH, MARK_DIM, generator=g),
        )
        for _ in range(n)
    ]


TRAIN = _batches(N_BATCHES, seed=1)
VAL = _batches(N_BATCHES, seed=2)
TEST = _batches(N_BATCHES, seed=3)
META = {"d_in": D_IN, "d_out": D_OUT}


def _loader_fn(**_kwargs):
    return TRAIN, VAL, TEST, META


def _exp_cfg(early_stopping: bool = True):
    """finetune_and_test only reads exp_cfg.eval_config; a stub keeps this focused.

    If it ever grows a dependency on another ExperimentConfig field, this raises
    AttributeError rather than silently testing the wrong thing.
    """
    eval_cfg = EvalConfig(
        task=_task(),
        datasets=[DatasetConfig(name="fake", loader_fn=_loader_fn, loader_kwargs={})],
        training_steps=1,
        extra_training_steps=MAX_STEPS,
        early_stopping=early_stopping,
        early_stop_checks=1,          # validate every step
        early_stop_min_delta=0.0,
        early_stop_patience=PATIENCE,  # finetune doubles this internally
        batch_size=BATCH_SIZE,
        optimizer="adam",
        optimizer_kwargs={"lr": 1e-3},
        device="cpu",
    )
    return SimpleNamespace(eval_config=eval_cfg)


def _initial_state():
    """A state_dict for the genome, distinct from whatever finetune trains to."""
    torch.manual_seed(1234)
    model = build_model_from_meta(_genome(), _task(), META)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def _test_mse_of(state: dict) -> float:
    """Independently evaluate `state` on TEST, mirroring finetune's own test loop."""
    model = build_model_from_meta(_genome(), _task(), META)
    model.load_state_dict(state, strict=True)
    model.eval()
    crit = nn.MSELoss()
    vals = []
    for x, y, x_mark, y_mark in TEST:
        y_hat = model(x, x_mark=x_mark, y_mark=y_mark)
        lc, dc = min(y_hat.size(1), y.size(1)), min(y_hat.size(2), y.size(2))
        vals.append(crit(y_hat[:, :lc, :dc], y[:, :lc, :dc]).item())
    return float(sum(vals) / len(vals))


# ----------------------------------------------------------------------------


def test_early_stopping_fires_when_validation_never_improves():
    """best_fitness=0.0 is unbeatable, so every check is a non-improvement."""
    torch.manual_seed(0)
    initial = _initial_state()

    metrics, _returned, _meta = finetune_and_test(
        best_fitness=0.0,
        genome=_genome(),
        exp_cfg=_exp_cfg(),
        initial_state_dict_cpu=initial,
        extra_training_steps=MAX_STEPS,
    )

    steps = metrics["finetune_steps_run"]
    assert steps < MAX_STEPS, (
        f"early stopping never fired: ran all {steps}/{MAX_STEPS} steps. "
        "patience is not being decremented on non-improvement"
    )
    # patience is doubled internally (early_stop_patience * 2), one check per step
    assert steps == PATIENCE * 2


def test_tested_model_matches_returned_state_when_no_improvement():
    """The model that produced test_mse must be the one whose weights we return."""
    torch.manual_seed(0)
    initial = _initial_state()

    metrics, returned, _meta = finetune_and_test(
        best_fitness=0.0,
        genome=_genome(),
        exp_cfg=_exp_cfg(),
        initial_state_dict_cpu=initial,
        extra_training_steps=MAX_STEPS,
    )

    assert returned is not None
    for k, v in initial.items():
        assert torch.equal(returned[k], v), f"returned state diverged from initial at {k}"

    # The real assertion: equality of the returned dict alone passed before the fix.
    # test_mse must match an independent evaluation of those same weights.
    expected = _test_mse_of(initial)
    assert metrics["test_mse"] == pytest.approx(expected, abs=1e-6), (
        f"test_mse={metrics['test_mse']:.6f} but the returned weights score "
        f"{expected:.6f} — the tested model is not the returned model"
    )


def test_no_initial_state_returns_the_weights_it_tested():
    """run_exp.py passes None when no best package was saved; don't return None."""
    torch.manual_seed(0)

    metrics, returned, _meta = finetune_and_test(
        best_fitness=0.0,
        genome=_genome(),
        exp_cfg=_exp_cfg(),
        initial_state_dict_cpu=None,
        extra_training_steps=MAX_STEPS,
    )

    assert returned is not None, "returned None, which run_exp.py would save as the checkpoint"
    expected = _test_mse_of(returned)
    assert metrics["test_mse"] == pytest.approx(expected, abs=1e-6)


def test_early_stopping_flag_is_honoured():
    """early_stopping=False must run the full budget instead of stopping early."""
    torch.manual_seed(0)
    initial = _initial_state()

    metrics, _returned, _meta = finetune_and_test(
        best_fitness=0.0,
        genome=_genome(),
        exp_cfg=_exp_cfg(early_stopping=False),
        initial_state_dict_cpu=initial,
        extra_training_steps=MAX_STEPS,
    )

    assert metrics["finetune_steps_run"] == MAX_STEPS
