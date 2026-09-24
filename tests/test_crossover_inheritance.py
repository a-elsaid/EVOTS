"""
Every genome-level spec must be heritable.

crossover_genome built the child Genome naming only the scalar fields and
stages, so var_head, cross_head, conv_block and quantum_block all fell back to
defaults and repair_genome re-randomised them. The search therefore resampled
those specs on every child instead of converging on them -- a pre-existing bug
affecting forecasting runs, not only the quantum work.

Each spec is chosen independently, so a child can take its var_head from one
parent and its conv_block from the other, exactly as the scalar fields behave.
"""

import dataclasses

import pytest

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.core.genome_v2 import Genome
from nas_ts.search.crossover import crossover_genome
from nas_ts.search.genome_init import random_genome
from nas_ts.search.repair import repair_genome
from nas_ts.utils.seeding import seed_everything

CONFIG = "configs/master_config.yml"          # forecasting: the space this affects

# spec -> (fields compared, family, tokenizer, block_type, parent1 values, parent2 values)
CASES = {
    "var_head": (
        ("encoder_type", "conv_kernel", "decomp_kernel"), "iT", "var", "attn",
        {"encoder_type": "linear", "conv_kernel": 3, "decomp_kernel": 3},
        {"encoder_type": "conv", "conv_kernel": 7, "decomp_kernel": 7},
    ),
    "cross_head": (
        ("groups", "patch_size", "stride", "encoder_type"), "Crossformer", "cross", "attn",
        {"groups": 1, "patch_size": 4, "stride": 1, "encoder_type": "linear"},
        {"groups": 1, "patch_size": 16, "stride": 8, "encoder_type": "conv"},
    ),
    "conv_block": (
        ("kernel_size", "dilation"), "iT", "var", "conv",
        {"kernel_size": 3, "dilation": 1},
        {"kernel_size": 7, "dilation": 4},
    ),
    "quantum_block": (
        ("encoding", "readout", "n_qubits", "reupload"), "iT", "var", "quantum",
        {"encoding": "amplitude", "readout": "state", "n_qubits": 5, "reupload": False},
        {"encoding": "angle", "readout": "expval_z", "n_qubits": 11, "reupload": True},
    ),
}


@pytest.fixture(scope="module")
def exp():
    return build_experiment(load_cfg(CONFIG, overrides=None))


def _space(exp):
    """Widen the ranges the test values need; master pins several to one option."""
    return dataclasses.replace(
        exp.search_space,
        conv_kernel_sizes=[3, 5, 7], conv_dilations=[1, 2, 4],
        quantum_amplitude_qubits_range=(4, 12), quantum_angle_qubits_range=(4, 12),
    )


def _parent(exp, ss, spec, seed, values):
    _, family, tok, bt, _, _ = CASES[spec]
    seed_everything(seed)
    g = random_genome(ss, exp.genome_constraints)
    g.family = family
    for st in g.stages:
        st.tokenizer = tok
        for b in st.blocks:
            b.block_type = bt
    g = repair_genome(g, ss)
    for k, v in values.items():
        setattr(getattr(g, spec), k, v)
    return g


def _sig(g, spec):
    return tuple(getattr(getattr(g, spec), f) for f in CASES[spec][0])


@pytest.mark.parametrize("spec", sorted(CASES))
def test_child_inherits_spec_from_one_parent(exp, spec):
    ss = _space(exp)
    _, _, _, _, v1, v2 = CASES[spec]
    p1, p2 = _parent(exp, ss, spec, 0, v1), _parent(exp, ss, spec, 1, v2)
    s1, s2 = _sig(p1, spec), _sig(p2, spec)
    assert s1 != s2, f"{spec}: parents are not distinct, the test proves nothing"

    counts = {"p1": 0, "p2": 0, "neither": 0}
    for i in range(30):
        seed_everything(900 + i)
        child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
        sig = _sig(child, spec)
        counts["p1" if sig == s1 else "p2" if sig == s2 else "neither"] += 1

    assert counts["neither"] == 0, (
        f"{spec}: {counts['neither']}/30 children matched neither parent — the spec "
        f"is being resampled, not inherited ({counts})")
    assert counts["p1"] > 0 and counts["p2"] > 0, (
        f"{spec}: children only ever came from one parent ({counts})")


@pytest.mark.parametrize("spec", sorted(CASES))
def test_child_does_not_alias_a_parents_spec(exp, spec):
    """A shared object would let a later mutation rewrite a parent in the population."""
    ss = _space(exp)
    _, _, _, _, v1, v2 = CASES[spec]
    p1, p2 = _parent(exp, ss, spec, 0, v1), _parent(exp, ss, spec, 1, v2)
    before1, before2 = _sig(p1, spec), _sig(p2, spec)

    seed_everything(11)
    child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
    assert getattr(child, spec) is not getattr(p1, spec)
    assert getattr(child, spec) is not getattr(p2, spec)

    field = CASES[spec][0][0]
    setattr(getattr(child, spec), field, "MUTATED")
    assert _sig(p1, spec) == before1, f"{spec}: mutating the child rewrote parent1"
    assert _sig(p2, spec) == before2, f"{spec}: mutating the child rewrote parent2"


def test_specs_are_chosen_independently(exp):
    """
    A child may take var_head from one parent and conv_block from the other. If
    all specs came from a single coin flip, the two would always agree.
    """
    ss = _space(exp)
    seed_everything(0)
    p1 = random_genome(ss, exp.genome_constraints)
    p1.family = "iT"
    for st in p1.stages:
        st.tokenizer = "var"
        for b in st.blocks:
            b.block_type = "conv"
    p1 = repair_genome(p1, ss)
    p1.var_head.encoder_type = "linear"; p1.conv_block.kernel_size = 3
    p2 = dataclasses.replace(p1)
    import copy as _copy
    p2 = _copy.deepcopy(p1)
    p2.var_head.encoder_type = "conv"; p2.conv_block.kernel_size = 7

    combos = set()
    for i in range(40):
        seed_everything(400 + i)
        c = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
        combos.add((c.var_head.encoder_type, c.conv_block.kernel_size))
    mixed = {c for c in combos if c in {("linear", 7), ("conv", 3)}}
    assert mixed, (
        f"no child mixed specs from different parents: {sorted(combos)} — the specs "
        f"appear to share one choice rather than being drawn independently")
