"""Regression tests for worker -> main weight transfer.

These cover three defects behind the reported symptom (main's finetune
validation error looks random-init instead of continuing from the search
winner's weights):

  1. PatchTokenHead / CrossTokenHead build their input projection inside
     forward(), so a freshly-rebuilt model does not own those keys and
     load_state_dict() rejects (or, with strict=False, silently drops) them.
  2. Because those layers do not exist when the optimizer is constructed,
     they are never registered and never trained.
  3. QuantumMixBlock stashes a tensorcircuit closure on self after the first
     forward pass, which makes the whole model unpicklable.

Everything here runs on CPU with random tensors in a couple of seconds.
"""

from __future__ import annotations

import math
import pickle

import pytest
import torch

from nas_ts.core.config import TaskConfig
from nas_ts.core.genome_v2 import (
    BlockSpec,
    CrossTokenHeadSpec,
    Genome,
    QuantumBlockSpec,
    StageSpec,
)
from nas_ts.models.model_builder_v2 import build_model


MODEL_DIM = 32
D_IN = 7
D_OUT = 7
INPUT_LENGTH = 96
PRED_LENGTH = 24
BATCH_SIZE = 4
NUM_BATCHES = 8

TOKENIZERS = ["time", "patch", "var", "cross"]


def _task(input_length: int = INPUT_LENGTH, pred_length: int = PRED_LENGTH) -> TaskConfig:
    return TaskConfig(
        task_type="forecasting",
        input_length=input_length,
        pred_length=pred_length,
        d_in=D_IN,
        d_out=D_OUT,
        metrics=["mse"],
        use_norm=False,
    )


def _genome(tokenizer: str, block_type: str = "attn") -> Genome:
    """Single stage, single block. dropout=0 keeps eval deterministic."""
    return Genome(
        family="test",
        model_dim=MODEL_DIM,
        num_heads=4,
        ff_mult=2.0,
        pos_encoding="none",
        dropout=0.0,
        cross_head=CrossTokenHeadSpec(
            enabled=True, groups=2, patch_size=8, stride=8, encoder_type="linear"
        ),
        stages=[StageSpec(name="stage0", tokenizer=tokenizer,
                          blocks=[BlockSpec(block_type=block_type)])],
    )


def _fake_batches(n: int, input_length: int = INPUT_LENGTH,
                  pred_length: int = PRED_LENGTH, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(BATCH_SIZE, input_length, D_IN, generator=g),
            torch.randn(BATCH_SIZE, pred_length, D_OUT, generator=g),
        )
        for _ in range(n)
    ]


def _align(y_hat: torch.Tensor, y: torch.Tensor):
    """Mirrors evaluate._align_pred_target."""
    lc = min(y_hat.size(1), y.size(1))
    dc = min(y_hat.size(2), y.size(2))
    return y_hat[:, :lc, :dc], y[:, :lc, :dc]


@torch.no_grad()
def _val_mse(model: torch.nn.Module, batches) -> float:
    model.eval()
    vals = []
    for x, y in batches:
        y_hat, y_aligned = _align(model(x), y)
        vals.append(torch.mean((y_hat - y_aligned) ** 2).item())
    return float(sum(vals) / len(vals))


def _train_briefly(model: torch.nn.Module, batches) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for x, y in batches:
        opt.zero_grad()
        y_hat, y_aligned = _align(model(x), y)
        torch.mean((y_hat - y_aligned) ** 2).backward()
        opt.step()


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_roundtrip_preserves_output(tokenizer):
    """Save -> rebuild from genome -> load(strict=True) must reproduce val MSE."""
    torch.manual_seed(0)
    task = _task()
    genome = _genome(tokenizer)

    train_batches = _fake_batches(NUM_BATCHES, seed=1)
    val_batches = _fake_batches(NUM_BATCHES // 2, seed=2)

    model = build_model(genome, task)
    _train_briefly(model, train_batches)
    mse_before = _val_mse(model, val_batches)

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.manual_seed(1234)  # different init, so a dropped key is visible
    rebuilt = build_model(genome, task)
    rebuilt.load_state_dict(state, strict=True)
    mse_after = _val_mse(rebuilt, val_batches)

    assert mse_after == pytest.approx(mse_before, abs=1e-6)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_optimizer_covers_all_params(tokenizer):
    """Every parameter that exists after a forward pass must be in the optimizer."""
    torch.manual_seed(0)
    model = build_model(_genome(tokenizer), _task())

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracked = {id(p) for group in opt.param_groups for p in group["params"]}

    x, _ = _fake_batches(1, seed=3)[0]
    model(x)

    untracked = [name for name, p in model.named_parameters() if id(p) not in tracked]
    assert untracked == [], f"parameters never registered with the optimizer: {untracked}"


def test_quantum_block_pickles_after_forward():
    """A quantum model must survive the process boundary after its circuit is built."""
    assert MODEL_DIM > 0 and (MODEL_DIM & (MODEL_DIM - 1)) == 0, "model_dim must be a power of 2"

    torch.manual_seed(0)
    short_len = 8  # keeps the tensorcircuit vmap tiny
    genome = _genome("time", block_type="quantum")
    genome.quantum_block = QuantumBlockSpec(
        enabled=True, nlayers=1, entangle_pattern="linear",
        gate_set="rx_ry", use_ffn=False,
    )

    model = build_model(genome, _task(input_length=short_len, pred_length=4))

    x, _ = _fake_batches(1, input_length=short_len, pred_length=4, seed=4)[0]
    model(x)  # builds and caches the circuit closure

    pickle.dumps(model)
