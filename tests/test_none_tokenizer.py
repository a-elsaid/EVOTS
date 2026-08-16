"""Tests for the `none` tokenizer (per-feature learned affine, no cross-feature mixing).

Covers the traps called out for this feature:
  1. Shape: [B, 1, d_in] -> tokens [B, d_in, D] -> logits [B, num_classes].
  2. Identity preservation: equal-valued features must NOT collapse into identical
     tokens (the defect `var`'s single shared Linear(1 -> D) has at seq_len=1).
  3. repair_genome must not reset a "none" stage back to "var"/"time" for family "iT".
  4. NoneTokenHead's weight/bias are built in __init__, so they round-trip through
     state_dict() / load_state_dict(strict=True) like every other tokenizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nas_ts.core.config import SearchSpaceConfig, TaskConfig
from nas_ts.core.genome_v2 import BlockSpec, Genome, StageSpec
from nas_ts.models.model_builder_v2 import (
    NoneTokenHead,
    VarTokenHead_iTransformer,
    build_model,
)
from nas_ts.search.repair import repair_genome


MODEL_DIM = 32
D_IN = 4          # Iris feature count
NUM_CLASSES = 3
BATCH_SIZE = 5


def _classification_task() -> TaskConfig:
    return TaskConfig(
        task_type="classification",
        input_length=1,
        pred_length=1,
        d_in=D_IN,
        d_out=D_IN,
        metrics=["loss", "accuracy"],
        use_norm=False,
    )


def _genome(tokenizer: str = "none") -> Genome:
    return Genome(
        family="iT",
        model_dim=MODEL_DIM,
        num_heads=4,
        ff_mult=2.0,
        pos_encoding="none",
        dropout=0.0,
        stages=[StageSpec(name="stage0", tokenizer=tokenizer,
                          blocks=[BlockSpec(block_type="attn")])],
    )


def _search_space(stage_tokenizers) -> SearchSpaceConfig:
    return SearchSpaceConfig(
        families=["iT"],
        depth_range=(2, 4),
        model_dim_range=(32, 64),
        num_heads_range=(4, 4),
        ff_mult_range=(4.0, 4.0),
        patch_sizes=[4],
        strides=[1],
        overlaps=[0.0],
        per_channel_options=[True],
        cross_groups=[1],
        cross_fusions=["add"],
        freq_types=["none"],
        freq_keep_ratios=[0.5],
        decomp_modes=["none"],
        decomp_kernel_sizes=[3],
        conv_kernel_sizes=[3],
        conv_dilations=[1],
        block_types=["attn"],
        pos_encoding_options=["none"],
        dropout_range=(0.0, 0.1),
        stage_tokenizers=stage_tokenizers,
    )


# ---------------------------------------------------------------------------


def test_none_tokenizer_shapes():
    torch.manual_seed(0)
    task = _classification_task()
    genome = _genome("none")

    model = build_model(genome, task, num_classes=NUM_CLASSES)

    x = torch.randn(BATCH_SIZE, 1, D_IN)

    tok = model.stage_modules[0]["tok"]
    assert isinstance(tok, NoneTokenHead)
    tokens = tok(x, x_mark=None)
    assert tokens.shape == (BATCH_SIZE, D_IN, MODEL_DIM)

    logits = model(x)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_none_tokenizer_preserves_feature_identity():
    torch.manual_seed(0)

    # All four features carry the same value: [B, 1, d_in], every entry identical.
    x = torch.full((BATCH_SIZE, 1, D_IN), 3.0)

    none_tok = NoneTokenHead(n_features=D_IN, d_model=MODEL_DIM)
    tokens = none_tok(x, x_mark=None)  # [B, d_in, D]

    # Per-feature tokens must differ from each other despite equal input values.
    first = tokens[:, 0, :]
    for j in range(1, D_IN):
        assert not torch.allclose(tokens[:, j, :], first), (
            f"feature {j}'s token matches feature 0's token despite per-feature "
            f"weights -- NoneTokenHead is not preserving feature identity"
        )

    # Contrast: var's single shared Linear(1 -> D) DOES collapse equal-valued
    # features into identical tokens at seq_len=1. Document that failure mode.
    var_tok = VarTokenHead_iTransformer(seq_len=1, d_model=MODEL_DIM, dropout=0.0)
    var_tokens = var_tok(x, x_mark=None)  # [B, d_in, D]
    var_first = var_tokens[:, 0, :]
    for j in range(1, D_IN):
        assert torch.allclose(var_tokens[:, j, :], var_first), (
            "expected var tokenizer to collapse equal-valued features into "
            "identical tokens (that's the bug 'none' fixes) -- if this fails, "
            "the contrast documented here is stale"
        )


def test_none_tokenizer_survives_repair():
    genome = _genome("none")
    ss = _search_space(stage_tokenizers=["var", "none"])

    repaired = repair_genome(genome, ss)

    assert repaired.stages[0].tokenizer == "none"


def test_none_tokenizer_state_dict_roundtrip():
    torch.manual_seed(0)
    task = _classification_task()
    genome = _genome("none")

    model = build_model(genome, task, num_classes=NUM_CLASSES)
    x = torch.randn(BATCH_SIZE, 1, D_IN)
    model(x)  # exercise a forward pass before snapshotting

    state = model.state_dict()
    assert "stage_modules.0.tok.weight" in state
    assert "stage_modules.0.tok.bias" in state
    assert state["stage_modules.0.tok.weight"].shape == (D_IN, MODEL_DIM)
    assert state["stage_modules.0.tok.bias"].shape == (D_IN, MODEL_DIM)

    torch.manual_seed(1234)  # different init, so a dropped/missing key would show up
    rebuilt = build_model(genome, task, num_classes=NUM_CLASSES)

    missing, unexpected = rebuilt.load_state_dict(state, strict=True)
    assert not missing
    assert not unexpected

    with torch.no_grad():
        assert torch.allclose(rebuilt(x), model(x))


def test_none_tokenizer_trains():
    """weight/bias must be real learnable Parameters that receive gradient and move."""
    torch.manual_seed(0)
    task = _classification_task()
    genome = _genome("none")

    model = build_model(genome, task, num_classes=NUM_CLASSES)
    tok = model.stage_modules[0]["tok"]
    assert isinstance(tok, NoneTokenHead)

    # 1) real Parameters, not buffers/plain tensors, and registered on the model.
    assert isinstance(tok.weight, nn.Parameter)
    assert isinstance(tok.bias, nn.Parameter)
    param_ids = {id(p) for p in model.parameters()}
    assert id(tok.weight) in param_ids
    assert id(tok.bias) in param_ids

    # 2) one forward + backward pass must produce non-null, non-zero gradients.
    x = torch.randn(BATCH_SIZE, 1, D_IN)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    assert tok.weight.grad is not None
    assert tok.bias.grad is not None
    assert torch.any(tok.weight.grad != 0)
    assert torch.any(tok.bias.grad != 0)

    # 3) an optimizer step must actually move the weight.
    weight_before = tok.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    assert not torch.allclose(tok.weight, weight_before)
