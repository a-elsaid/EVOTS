import random
from typing import List, Optional

from ..core.config import SearchSpaceConfig, GenomeConstraints
from ..core.genome_v2 import Genome, StageSpec, BlockSpec
from .repair import repair_genome


def random_block(ss: SearchSpaceConfig) -> BlockSpec:
    return BlockSpec(
        block_type=random.choice(ss.block_types),
        dim=None,
        num_heads=None,
        ff_mult=None,
        norm="layernorm",
    )


def _random_stages(ss: SearchSpaceConfig, depth: int, family: str) -> List[StageSpec]:
    """Build a random single- or multi-stage pipeline respecting family constraints."""
    min_stages, max_stages = getattr(ss, "stage_count_range")
    num_stages = random.randint(int(min_stages), int(max_stages))

    per_stage = [1] * num_stages
    remaining = max(0, depth - num_stages)
    for _ in range(remaining):
        per_stage[random.randrange(num_stages)] += 1

    if family == "iT":
        tok_choices = ["var"]
    elif family == "PatchTST":
        tok_choices = ["patch"]
    elif family == "Crossformer":
        tok_choices = ["cross"]
    else:
        tok_choices = ["time", "var", "patch", "cross"]

    stages: List[StageSpec] = []
    prev_tok: Optional[str] = None

    for i in range(num_stages):
        tokenizer = random.choice(tok_choices)

        # discourage repeating the same tokenizer back-to-back (Hybrid only)
        if prev_tok is not None and tokenizer == prev_tok and len(tok_choices) > 1:
            tokenizer = random.choice([t for t in tok_choices if t != prev_tok])

        if i == 0:
            retokenize = "none"
        else:
            retok_choices = getattr(ss, "stage_retokens", ["none", "cross_attn"])
            retokenize = random.choice(retok_choices)

        stages.append(StageSpec(
            name=f"stage{i}",
            tokenizer=tokenizer,
            retokenize=retokenize,
            blocks=[random_block(ss) for _ in range(per_stage[i])],
        ))
        prev_tok = tokenizer

    return stages


def random_genome(
    ss: SearchSpaceConfig,
    constraints: Optional[GenomeConstraints] = None,
) -> Genome:
    if constraints is None:
        min_blocks, max_blocks = ss.depth_range
    else:
        min_blocks = max(constraints.min_blocks, ss.depth_range[0])
        max_blocks = min(constraints.max_blocks, ss.depth_range[1])

    depth = random.randint(min_blocks, max_blocks)
    family = random.choice(ss.families)

    g = Genome(
        family=family,
        model_dim=random.randint(*ss.model_dim_range),
        num_heads=random.randint(*ss.num_heads_range),
        ff_mult=random.uniform(*ss.ff_mult_range),
        pos_encoding=random.choice(ss.pos_encoding_options),
        dropout=random.uniform(*ss.dropout_range),
        stages=_random_stages(ss, depth, family),
    )

    return repair_genome(g, ss)