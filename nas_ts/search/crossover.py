from __future__ import annotations

import copy
import random
from typing import Optional, List, Tuple

from ..core.config import SearchSpaceConfig, GenomeConstraints, EvolutionConfig
from ..core.genome_v2 import Genome, StageSpec, BlockSpec
from .repair import repair_genome


def _stage_bounds(ss: SearchSpaceConfig) -> Tuple[int, int]:
    mn, mx = getattr(ss, "stage_count_range", (1, 3))
    mn = max(1, int(mn))
    mx = max(mn, int(mx))
    return mn, mx


def _ensure_min_stage(stages: List[StageSpec]) -> List[StageSpec]:
    if stages:
        return stages
    return [StageSpec(name="stage0", tokenizer="time", retokenize="none", blocks=[BlockSpec("attn")])]


def _ensure_min_blocks_per_stage(stages: List[StageSpec]) -> None:
    for st in stages:
        if not st.blocks:
            st.blocks = [BlockSpec("attn")]


def _renumber_stages(stages: List[StageSpec]) -> None:
    for i, st in enumerate(stages):
        st.name = f"stage{i}"


def _total_blocks(stages: List[StageSpec]) -> int:
    return sum(len(st.blocks or []) for st in stages)


def _splice_blocks(b1: List[BlockSpec], b2: List[BlockSpec]) -> List[BlockSpec]:
    """Prefix of b1 + suffix of b2."""
    b1 = list(b1 or [])
    b2 = list(b2 or [])
    if not b1 and not b2:
        return [BlockSpec("attn")]
    if not b1:
        return copy.deepcopy(b2)
    if not b2:
        return copy.deepcopy(b1)

    c1 = random.randint(0, len(b1))
    c2 = random.randint(0, len(b2))
    out = copy.deepcopy(b1[:c1]) + copy.deepcopy(b2[c2:])
    return out if out else [BlockSpec("attn")]


def _pick_stage_template(s1: List[StageSpec], s2: List[StageSpec], idx: int) -> StageSpec:
    """Pick a stage template (tokenizer/retokenize) from one of the parents at position idx."""
    has1 = idx < len(s1)
    has2 = idx < len(s2)

    if has1 and has2:
        return copy.deepcopy(random.choice([s1[idx], s2[idx]]))
    if has1:
        return copy.deepcopy(s1[idx])
    if has2:
        return copy.deepcopy(s2[idx])
    return copy.deepcopy(random.choice(s1 + s2))


def _make_child_stages(
    p1_stages: List[StageSpec],
    p2_stages: List[StageSpec],
    ss: SearchSpaceConfig,
    constraints: Optional[GenomeConstraints],
) -> List[StageSpec]:
    """Build child stages by splicing blocks from corresponding parent stages."""
    s1 = _ensure_min_stage(p1_stages)
    s2 = _ensure_min_stage(p2_stages)

    min_stages, max_stages = _stage_bounds(ss)
    target_stages = random.randint(min_stages, max_stages)

    child: List[StageSpec] = []
    for i in range(target_stages):
        tmpl = _pick_stage_template(s1, s2, i)
        b1 = s1[i].blocks if i < len(s1) else random.choice(s1).blocks
        b2 = s2[i].blocks if i < len(s2) else random.choice(s2).blocks
        tmpl.blocks = _splice_blocks(b1 or [], b2 or [])
        child.append(tmpl)

    _ensure_min_blocks_per_stage(child)
    _renumber_stages(child)

    if constraints is not None:
        while _total_blocks(child) > constraints.max_blocks:
            last = child[-1]
            if len(last.blocks) > 1:
                last.blocks.pop()
            elif len(child) > min_stages:
                child.pop()
            else:
                break
            if not child:
                child = _ensure_min_stage([])
            _ensure_min_blocks_per_stage(child)

        while _total_blocks(child) < constraints.min_blocks:
            random.choice(child).blocks.append(BlockSpec(random.choice(getattr(ss, "block_types", ["attn"]))))
            _ensure_min_blocks_per_stage(child)

        # Enforce stage count bounds after block trimming
        while len(child) < min_stages:
            st = _pick_stage_template(s1, s2, len(child))
            st.blocks = [BlockSpec(random.choice(getattr(ss, "block_types", ["attn"])))]
            child.append(st)
            _ensure_min_blocks_per_stage(child)
            _renumber_stages(child)

        while len(child) > max_stages and len(child) > min_stages:
            child.pop()
            _ensure_min_blocks_per_stage(child)
            _renumber_stages(child)

    return child


def crossover_genome(
    parent1: Genome,
    parent2: Genome,
    ss: SearchSpaceConfig,
    evo: EvolutionConfig,
    constraints: Optional[GenomeConstraints] = None,
) -> Genome:
    """
    Stage+block crossover: child inherits stage structure and spliced blocks from
    both parents, with global hyperparams chosen randomly from either parent.
    Always finishes with repair_genome() to enforce invariants.
    """
    child_stages = _make_child_stages(
        list(parent1.stages or []),
        list(parent2.stages or []),
        ss,
        constraints,
    )

    child = Genome(
        family=str(random.choice([parent1.family, parent2.family])),
        model_dim=int(random.choice([parent1.model_dim, parent2.model_dim])),
        num_heads=int(random.choice([parent1.num_heads, parent2.num_heads])),
        ff_mult=float(random.choice([parent1.ff_mult, parent2.ff_mult])),
        pos_encoding=str(random.choice([parent1.pos_encoding, parent2.pos_encoding])),
        dropout=float(random.choice([parent1.dropout, parent2.dropout])),
        stages=child_stages,
    )

    return repair_genome(child, ss)