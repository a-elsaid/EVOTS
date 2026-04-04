from __future__ import annotations
import random

from ..core.config import SearchSpaceConfig
from ..core.genome_v2 import Genome, StageSpec, BlockSpec


def _round_pow2(x: int) -> int:
    if x <= 1:
        return 1
    lower = 1 << (x.bit_length() - 1)
    upper = lower << 1
    return lower if (x - lower) <= (upper - x) else upper


def _random_block(ss: SearchSpaceConfig) -> BlockSpec:
    return BlockSpec(
        block_type=random.choice(ss.block_types),
        dim=None,
        num_heads=None,
        ff_mult=None,
        norm="layernorm",
    )


def _allowed_tokenizers_for_family(family: str) -> set[str]:
    if family == "iT":
        return {"var"}
    if family == "PatchTST":
        return {"patch"}
    if family == "Crossformer":
        return {"cross"}
    return {"time", "var", "patch", "cross"}


def repair_genome(genome: Genome, ss: SearchSpaceConfig) -> Genome:
    """
    Enforces structural invariants on a v2 genome:
      - stages exists and has >= 1 stage
      - stage0.retokenize == "none"
      - each stage has >= 1 valid BlockSpec
      - tokenizers respect family constraints and search space
      - var_head / cross_head enabled iff used by a stage
      - model_dim rounded to nearest power of 2, clamped to range
      - num_heads derived from model_dim (head_dim = 8)
      - ff_mult, dropout clamped to range
      - head params clamped to search space
    """

    # 1) Ensure stages exist
    if not genome.stages:
        genome.stages = [
            StageSpec(
                name="stage0",
                tokenizer="time",
                retokenize="none",
                blocks=[_random_block(ss)],
            )
        ]

    # 2) Fix invalid stage fields
    valid_tokenizers = set(getattr(ss, "stage_tokenizers", ["time", "var", "patch", "cross"]))
    valid_retok = set(getattr(ss, "stage_retokens", ["none", "cross_attn"]))

    for i, st in enumerate(genome.stages):
        if st.tokenizer not in valid_tokenizers:
            st.tokenizer = "time"

        st.retokenize = "none" if i == 0 else (st.retokenize if st.retokenize in valid_retok else "none")

        if not st.blocks:
            st.blocks = [_random_block(ss)]

        st.blocks = [b if isinstance(b, BlockSpec) else _random_block(ss) for b in st.blocks]

        if not st.name:
            st.name = f"stage{i}"

    # 3) Enforce family -> tokenizer constraints
    allowed = _allowed_tokenizers_for_family(genome.family)
    for st in genome.stages:
        if st.tokenizer not in allowed:
            st.tokenizer = next(iter(allowed))

    # 4) Enable heads based on actual stage usage
    genome.var_head.enabled = any(st.tokenizer == "var" for st in genome.stages)
    genome.cross_head.enabled = any(st.tokenizer == "cross" for st in genome.stages)

    # 5) Clamp and round model_dim, derive num_heads
    if hasattr(ss, "model_dim_range"):
        lo, hi = ss.model_dim_range
        D = _round_pow2(int(max(lo, min(hi, genome.model_dim))))
        genome.model_dim = int(max(lo, min(hi, D)))

    derived_heads = max(1, genome.model_dim // 8)
    if hasattr(ss, "num_heads_range"):
        hlo, hhi = ss.num_heads_range
        genome.num_heads = int(max(hlo, min(hhi, derived_heads)))
    else:
        genome.num_heads = int(derived_heads)

    if hasattr(ss, "ff_mult_range"):
        lo, hi = ss.ff_mult_range
        genome.ff_mult = float(max(lo, min(hi, genome.ff_mult)))

    if hasattr(ss, "dropout_range"):
        lo, hi = ss.dropout_range
        genome.dropout = float(max(lo, min(hi, genome.dropout)))

    # 6) Repair var_head params
    if genome.var_head.enabled:
        if hasattr(ss, "var_encoder_types") and genome.var_head.encoder_type not in ss.var_encoder_types:
            genome.var_head.encoder_type = random.choice(list(ss.var_encoder_types))

        if hasattr(ss, "var_conv_kernels") and genome.var_head.conv_kernel not in ss.var_conv_kernels:
            genome.var_head.conv_kernel = random.choice(list(ss.var_conv_kernels))

        genome.var_head.fft_keep_ratio = max(0.01, min(1.0, float(genome.var_head.fft_keep_ratio)))

        if hasattr(ss, "var_decomp_kernels") and genome.var_head.decomp_kernel not in ss.var_decomp_kernels:
            genome.var_head.decomp_kernel = random.choice(list(ss.var_decomp_kernels))

    # 7) Repair cross_head params
    if genome.cross_head.enabled:
        if hasattr(ss, "cross_groups") and genome.cross_head.groups not in ss.cross_groups:
            genome.cross_head.groups = random.choice(list(ss.cross_groups))

        if hasattr(ss, "cross_patch_sizes") and genome.cross_head.patch_size not in ss.cross_patch_sizes:
            genome.cross_head.patch_size = random.choice(list(ss.cross_patch_sizes))

        if hasattr(ss, "cross_strides") and genome.cross_head.stride not in ss.cross_strides:
            genome.cross_head.stride = random.choice(list(ss.cross_strides))

        genome.cross_head.patch_size = max(1, int(genome.cross_head.patch_size))
        genome.cross_head.stride = max(1, int(genome.cross_head.stride))

        if hasattr(ss, "cross_encoder_types") and genome.cross_head.encoder_type not in ss.cross_encoder_types:
            genome.cross_head.encoder_type = random.choice(list(ss.cross_encoder_types))

        if hasattr(ss, "cross_conv_kernels") and genome.cross_head.conv_kernel not in ss.cross_conv_kernels:
            genome.cross_head.conv_kernel = random.choice(list(ss.cross_conv_kernels))

        if hasattr(ss, "cross_pools") and genome.cross_head.pool not in ss.cross_pools:
            genome.cross_head.pool = random.choice(list(ss.cross_pools))

    return genome