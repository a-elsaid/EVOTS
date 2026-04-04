#!/usr/bin/env python3
from ..core.genome_v2 import Genome, StageSpec, BlockSpec, PatchingSpec

def make_iTransformer_like(model_dim: int = 32, num_layers: int = 2) -> Genome:
    blocks = [BlockSpec(block_type="inv_attn") for _ in range(num_layers)]
    stages = [StageSpec(name="stage0", tokenizer="var", retokenize="cross_attn", blocks=blocks)]
    stages[0].retokenize = "none"  # enforce for stage0

    g = Genome(
        family="iT",
        model_dim=model_dim,
        num_heads=32,          # choose something compatible
        ff_mult=4.0,
        pos_encoding="absolute",
        dropout=0.1,
        stages=stages,
    )

    g.var_head.enabled = True
    return g


def make_PatchTST_like(model_dim: int = 128, num_layers: int = 4) -> Genome:
    blocks = [BlockSpec(block_type="attn") for _ in range(num_layers)]
    stages = [StageSpec(name="stage0", tokenizer="patch", retokenize="none", blocks=blocks)]
    return Genome(
        family="PatchTST",
        model_dim=model_dim,
        num_heads=32,          # or choose something compatible
        ff_mult=4.0,
        pos_encoding="absolute",
        dropout=0.1,
        stages=stages,
    )