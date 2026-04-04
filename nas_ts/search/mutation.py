from __future__ import annotations

import copy
import random
from typing import Optional

from ..core.config import SearchSpaceConfig, GenomeConstraints, EvolutionConfig
from ..core.genome_v2 import Genome, StageSpec, BlockSpec
from .repair import repair_genome


# -------------------------
# helpers
# -------------------------
def _random_block(ss: SearchSpaceConfig) -> BlockSpec:
    bt = random.choice(ss.block_types)
    return BlockSpec(block_type=bt, dim=None, num_heads=None, ff_mult=None, norm="layernorm")


def _random_stage(ss: SearchSpaceConfig, name: str, *, is_stage0: bool = False) -> StageSpec:
    # Use config-driven choices (falls back safely if missing)
    tok_choices = getattr(ss, "stage_tokenizers", ["time", "var", "patch", "cross"])
    retok_choices = getattr(ss, "stage_retokens", ["none", "cross_attn"])

    tokenizer = random.choice(tok_choices)

    # Stage0 should not retokenize; repair() will enforce tool
    # (keept simple here to avoid combinatorial explosion in mutation)
    if is_stage0:
        retokenize = "none"
    else:
        retokenize = random.choice(retok_choices)

    blocks = [_random_block(ss)]
    return StageSpec(name=name, tokenizer=tokenizer, retokenize=retokenize, blocks=blocks)


def _maybe_randomize_var_head(g: Genome, ss: SearchSpaceConfig) -> None:
    """If a stage uses tokenizer='var', ensure var_head is enabled + has valid params."""
    g.var_head.enabled = True

    enc_choices = list(getattr(ss, "var_encoder_types", ["linear", "conv", "fft", "decomp_linear"]))
    g.var_head.encoder_type = random.choice(enc_choices)

    if g.var_head.encoder_type == "conv":
        kernels = list(getattr(ss, "var_conv_kernels", [3, 5, 7]))
        g.var_head.conv_kernel = random.choice(kernels)
        g.var_head.conv_stride = 1 # conv_stride's less sensitive, kept fixed to reduce search space
        g.var_head.conv_pool = random.choice(["avg", "max"])

    elif g.var_head.encoder_type == "fft":
        keep = list(getattr(ss, "var_fft_keep_ratios", [0.125, 0.25, 0.5]))
        g.var_head.fft_keep_ratio = float(random.choice(keep))

    elif g.var_head.encoder_type == "decomp_linear":
        ks = list(getattr(ss, "var_decomp_kernels", [3, 5, 7]))
        g.var_head.decomp_kernel = int(random.choice(ks))


def _maybe_randomize_cross_head(g: Genome, ss: SearchSpaceConfig) -> None:
    """If a stage uses tokenizer='cross', ensure cross_head is enabled + has valid params."""
    g.cross_head.enabled = True

    g.cross_head.groups = int(getattr(g.cross_head, "groups", 2))

    patch_sizes = list(getattr(ss, "cross_patch_sizes", [4, 8]))
    strides = list(getattr(ss, "cross_strides", [1, 2]))
    enc_types = list(getattr(ss, "cross_encoder_types", ["linear", "conv"]))
    conv_kernels = list(getattr(ss, "cross_conv_kernels", [3, 5]))
    pools = list(getattr(ss, "cross_pools", ["avg", "max"]))

    g.cross_head.patch_size = int(random.choice(patch_sizes))
    g.cross_head.stride = int(random.choice(strides))
    g.cross_head.encoder_type = random.choice(enc_types)

    if g.cross_head.encoder_type == "conv":
        g.cross_head.conv_kernel = int(random.choice(conv_kernels))
        g.cross_head.pool = random.choice(pools)


# -------------------------
# main mutation
# -------------------------
def mutate_genome(
    genome: Genome,
    ss: SearchSpaceConfig,
    evo: EvolutionConfig,
    constraints: Optional[GenomeConstraints] = None,
) -> Genome:
    """
    Stage-native mutation for Genome v2.
    - Mutates global params occasionally
    - Mutates stages: add/remove stage, change tokenizer, toggle retokenize, edit blocks
    - When tokenizer requires a head ("var"/"cross"), enables + randomizes the head spec.
    - Always ends with repair_genome() to enforce invariants.
    """
    g = copy.deepcopy(genome)
    mr = float(getattr(evo, "mutation_rate", 0.3))

    # -------------------------
    # global mutations
    # -------------------------
    if random.random() < mr and hasattr(ss, "model_dim_range"):
        g.model_dim = random.randint(*ss.model_dim_range)

    if random.random() < mr and hasattr(ss, "num_heads_range"):
        g.num_heads = random.randint(*ss.num_heads_range)

    if random.random() < mr and hasattr(ss, "ff_mult_range"):
        lo, hi = ss.ff_mult_range
        g.ff_mult = random.uniform(lo, hi)

    if random.random() < mr and hasattr(ss, "dropout_range"):
        lo, hi = ss.dropout_range
        g.dropout = random.uniform(lo, hi)

    # family mutation is optional --> comment to keep stable unless explicitly wanted
    # if random.random() < mr:
    #     g.family = random.choice(ss.families)

    # -------------------------
    # stage mutations
    # -------------------------
    if g.stages is None:
        g.stages = []

    if len(g.stages) == 0:
        g.stages.append(_random_stage(ss, "stage0", is_stage0=True))

    min_stages, max_stages = getattr(ss, "stage_count_range")
    min_stages = max(1, int(min_stages))
    max_stages = max(min_stages, int(max_stages))

    if random.random() < mr:
        r = random.random()
        if r < 0.5 and len(g.stages) < max_stages:
            g.stages.append(_random_stage(ss, f"stage{len(g.stages)}", is_stage0=False))
        elif r >= 0.5 and len(g.stages) > min_stages:
            # remove non-stage0
            idx = random.randrange(1, len(g.stages))
            g.stages.pop(idx)

    st = random.choice(g.stages)

    tok_choices = getattr(ss, "stage_tokenizers", ["time", "var", "patch", "cross"])
    retok_choices = getattr(ss, "stage_retokens", ["none", "cross_attn"])

    if random.random() < mr:
        st.tokenizer = random.choice(tok_choices)

        # If tokenizer implies head, randomize that head
        if st.tokenizer == "var":
            _maybe_randomize_var_head(g, ss)
        elif st.tokenizer == "cross":
            _maybe_randomize_cross_head(g, ss)

    if random.random() < mr:
        if st is g.stages[0]:
            st.retokenize = "none"
        else:
            st.retokenize = random.choice(retok_choices)

    if st.blocks is None:
        st.blocks = [_random_block(ss)]

    if random.random() < mr:
        st.blocks.append(_random_block(ss))

    if random.random() < mr and len(st.blocks) > 1:
        st.blocks.pop(random.randrange(len(st.blocks)))

    if random.random() < mr and len(st.blocks) > 0:
        b = random.choice(st.blocks)
        b.block_type = random.choice(ss.block_types)

    # -------------------------
    # constraints on total blocks across stages
    # -------------------------
    if constraints is not None:
        total = sum(len(s.blocks) for s in g.stages)
        while total > constraints.max_blocks and len(g.stages) >= 1:
            last = g.stages[-1]
            if len(last.blocks) > 1:
                last.blocks.pop()
            elif len(g.stages) > 1:
                g.stages.pop()
            else:
                # single stage0: can't remove its last block
                break
            total = sum(len(s.blocks) for s in g.stages)

        # ensure minimum blocks
        total = sum(len(s.blocks) for s in g.stages)
        while total < constraints.min_blocks:
            random.choice(g.stages).blocks.append(_random_block(ss))
            total += 1

    # -------------------------
    # final repair (enforces invariants + enables heads if needed)
    # -------------------------
    g = repair_genome(g, ss)
    return g