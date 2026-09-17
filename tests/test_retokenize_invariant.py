"""Regression tests for issue #3: orphaned stages.

Inter-stage information only flows through the cross-attention retokenizer, and
the forecast head reads only the last stage's tokens.  A stage i>=1 with
retokenize == "none" rebuilds from raw x and discards prev_tokens, which orphans
every earlier stage: they are built and trained but cannot influence the output.

The fix makes retokenize position-determined rather than searched:
stage 0 -> "none" (no predecessor), every stage i>=1 -> "cross_attn".
These tests pin that invariant at three levels — the genome after repair, the
built module tree, and actual gradient flow through stage 0.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from nas_ts.core.config import (
    EvolutionConfig,
    GenomeConstraints,
    SearchSpaceConfig,
    TaskConfig,
)
from nas_ts.core.genome_v2 import BlockSpec, Genome, StageSpec
from nas_ts.models.model_builder_v2 import (
    CrossAttentionRetokenizer,
    StagedForecastModel,
    build_model,
)
from nas_ts.search.crossover import crossover_genome
from nas_ts.search.genome_init import random_genome
from nas_ts.search.mutation import mutate_genome
from nas_ts.search.repair import repair_genome


MODEL_DIM = 32
D_IN = 7
D_OUT = 7
INPUT_LENGTH = 32
PRED_LENGTH = 8
BATCH_SIZE = 4


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


def _search_space() -> SearchSpaceConfig:
    return SearchSpaceConfig(
        families=["Hybrid"],
        depth_range=(2, 6),
        model_dim_range=(16, 64),
        num_heads_range=(1, 8),
        ff_mult_range=(2.0, 4.0),
        patch_sizes=[4, 8],
        strides=[1, 2],
        overlaps=[0.0],
        per_channel_options=[True],
        cross_groups=[1, 2],
        cross_fusions=["add"],
        freq_types=["none"],
        freq_keep_ratios=[0.5],
        decomp_modes=["none"],
        decomp_kernel_sizes=[3],
        conv_kernel_sizes=[3],
        conv_dilations=[1],
        block_types=["attn", "inv_attn", "conv"],
        pos_encoding_options=["none"],
        dropout_range=(0.0, 0.1),
        stage_count_range=(1, 4),
    )


def _genome(num_stages: int, retokenize: str = "none") -> Genome:
    """Multi-stage genome, all stages 'time'-tokenized so shapes line up."""
    return Genome(
        family="Hybrid",
        model_dim=MODEL_DIM,
        num_heads=4,
        ff_mult=2.0,
        pos_encoding="none",
        dropout=0.0,
        stages=[
            StageSpec(
                name=f"stage{i}",
                tokenizer="time",
                retokenize=retokenize,
                blocks=[BlockSpec(block_type="attn")],
            )
            for i in range(num_stages)
        ],
    )


# ---------------------------------------------------------------------------


def test_repair_forces_cross_attn_on_later_stages():
    """A K=3 genome seeded entirely with "none" comes out position-determined."""
    genome = _genome(3, retokenize="none")

    repair_genome(genome, _search_space())

    assert genome.stages[0].retokenize == "none"
    assert genome.stages[1].retokenize == "cross_attn"
    assert genome.stages[2].retokenize == "cross_attn"


def test_built_model_retokenizers_are_index_determined():
    """Stage 0 gets Identity; every later stage gets a real retokenizer."""
    torch.manual_seed(0)
    model = build_model(_genome(2), _task())

    assert isinstance(model, StagedForecastModel)
    assert isinstance(model.stage_modules[0]["retok"], nn.Identity)
    assert isinstance(model.stage_modules[1]["retok"], CrossAttentionRetokenizer)


def test_built_model_ignores_stale_retokenize_field():
    """An un-repaired genome (seed / loaded JSON) still builds the invariant."""
    torch.manual_seed(0)
    # stage1 says "none" — the builder must not honour it, or stage0 is orphaned.
    model = build_model(_genome(2, retokenize="none"), _task())

    assert isinstance(model.stage_modules[0]["retok"], nn.Identity)
    assert isinstance(model.stage_modules[1]["retok"], CrossAttentionRetokenizer)


def test_stage0_is_not_orphaned():
    """Gradient must reach stage 0 — proof it can influence the forecast."""
    torch.manual_seed(0)
    model = build_model(_genome(2), _task())
    model.train()

    x = torch.randn(BATCH_SIZE, INPUT_LENGTH, D_IN)
    model(x).sum().backward()

    grads = {
        name: p.grad
        for name, p in model.stage_modules[0].named_parameters()
        if p.requires_grad
    }
    assert grads, "stage 0 has no trainable parameters"

    live = [
        name for name, g in grads.items()
        if g is not None and torch.any(g != 0)
    ]
    assert live, (
        "no parameter in stage 0 received a non-zero gradient — stage 0 is "
        "orphaned and cannot influence the output (issue #3)"
    )


def test_search_never_produces_orphaned_stages():
    """Init + mutation + crossover, repaired, never leave a stage i>=1 at "none"."""
    ss = _search_space()
    evo = EvolutionConfig(
        population_size=10,
        max_evals=10,
        init_seed_fraction=0.0,
        mutation_rate=0.9,  # high, so stage add/remove fires often
        crossover_rate=1.0,
    )
    constraints = GenomeConstraints(min_blocks=2, max_blocks=8)

    def _check(g: Genome, origin: str) -> None:
        assert g.stages[0].retokenize == "none", f"{origin}: stage0 must not retokenize"
        for i, st in enumerate(g.stages[1:], start=1):
            assert st.retokenize == "cross_attn", (
                f"{origin}: stage{i} has retokenize={st.retokenize!r} — "
                "it would orphan every earlier stage"
            )

    random.seed(1234)
    for _ in range(40):
        g = random_genome(ss, constraints)
        _check(g, "init")

        for _ in range(5):
            g = mutate_genome(g, ss, evo, constraints)
            _check(g, "mutation")

        other = random_genome(ss, constraints)
        child = crossover_genome(g, other, ss, evo, constraints)
        _check(child, "crossover")
