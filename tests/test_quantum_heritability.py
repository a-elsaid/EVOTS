"""
Quantum genes must survive mutation and crossover.

Crossover built the child Genome without quantum_block, so the child got a
default spec and repair_genome re-randomised it: two parents that agreed on a
circuit produced children that did not. Evolution cannot converge on a circuit it
cannot transmit, and the search would have looked like it was exploring circuits
while actually resampling them every generation.
"""

import dataclasses

import pytest

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.core.genome_v2 import Genome
from nas_ts.search.crossover import crossover_genome
from nas_ts.search.genome_init import random_genome
from nas_ts.search.mutation import _maybe_randomize_quantum_block, mutate_genome
from nas_ts.search.repair import repair_genome
from nas_ts.utils.seeding import seed_everything

CONFIG = "configs/classification/iris.yml"
GENES = ("encoding", "readout", "n_qubits", "reupload")


@pytest.fixture(scope="module")
def exp():
    return build_experiment(load_cfg(CONFIG, overrides=None))


def _space(exp, **kw):
    return dataclasses.replace(exp.search_space, **kw)


def _all_quantum(g):
    for st in g.stages:
        for b in st.blocks:
            b.block_type = "quantum"
    g.quantum_block.enabled = True
    return g


def _parent(exp, ss, seed, **genes):
    seed_everything(seed)
    g = _all_quantum(random_genome(ss, exp.genome_constraints))
    g = repair_genome(g, ss)
    for k, v in genes.items():
        setattr(g.quantum_block, k, v)
    return g


def _sig(g):
    return tuple(getattr(g.quantum_block, gene) for gene in GENES)


# ------------------------------------------------------------------ mutation

def test_mutation_randomises_all_four_genes(exp):
    ss = _space(exp, quantum_encodings=["angle"], quantum_readouts=["expval_z"],
                quantum_angle_qubits_range=(6, 6), quantum_reupload_options=[True])
    g = _parent(exp, ss, 0, encoding="amplitude", readout="state",
                n_qubits=0, reupload=False)
    _maybe_randomize_quantum_block(g, ss)

    assert g.quantum_block.encoding == "angle"
    assert g.quantum_block.readout == "expval_z"
    assert g.quantum_block.n_qubits == 6
    assert g.quantum_block.reupload is True


def test_mutation_never_leaves_angle_unbuildable(exp):
    """Mutation must not depend on repair running afterwards to be valid."""
    ss = _space(exp, quantum_encodings=["angle"], quantum_angle_qubits_range=(4, 9))
    for seed in range(15):
        g = _parent(exp, ss, seed, n_qubits=0)
        _maybe_randomize_quantum_block(g, ss)
        assert g.quantum_block.n_qubits >= 4


def test_mutation_draws_width_from_the_chosen_encodings_range(exp):
    """The two encodings have separate ranges; the draw must follow the choice."""
    ss = _space(exp, quantum_encodings=["amplitude"],
                quantum_amplitude_qubits_range=(4, 5),
                quantum_angle_qubits_range=(12, 14))
    for seed in range(10):
        g = _parent(exp, ss, seed)
        _maybe_randomize_quantum_block(g, ss)
        assert g.quantum_block.encoding == "amplitude"
        assert 4 <= g.quantum_block.n_qubits <= 5, "drew from the angle range"


def test_mutate_genome_keeps_quantum_genes_valid(exp):
    """End to end through the real operator, repeatedly."""
    ss = _space(exp, quantum_amplitude_qubits_range=(4, 9),
                quantum_angle_qubits_range=(4, 9))
    g = _parent(exp, ss, 1)
    for _ in range(25):
        g = mutate_genome(g, ss, exp.evolution, exp.genome_constraints)
        q = g.quantum_block
        if not q.enabled:
            continue
        assert q.encoding in ss.quantum_encodings
        assert q.readout in ss.quantum_readouts
        if q.encoding == "angle":
            assert q.n_qubits > 0


# ----------------------------------------------------------------- crossover

def test_child_inherits_quantum_genes_from_one_parent(exp):
    """
    The load-bearing test. Two parents with deliberately distinct circuits: each
    child must match one of them, not a fresh random draw.
    """
    ss = _space(exp, quantum_amplitude_qubits_range=(4, 12),
                quantum_angle_qubits_range=(4, 12))
    p1 = _parent(exp, ss, 0, encoding="amplitude", readout="state",
                 n_qubits=5, reupload=False, nlayers=1)
    p2 = _parent(exp, ss, 1, encoding="angle", readout="expval_z",
                 n_qubits=11, reupload=True, nlayers=3)
    s1, s2 = _sig(p1), _sig(p2)
    assert s1 != s2

    counts = {"p1": 0, "p2": 0, "neither": 0}
    for i in range(40):
        seed_everything(100 + i)
        child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
        sig = _sig(child)
        counts["p1" if sig == s1 else "p2" if sig == s2 else "neither"] += 1

    assert counts["neither"] == 0, (
        f"{counts['neither']}/40 children matched neither parent: genes are being "
        f"resampled, not inherited ({counts})")
    assert counts["p1"] > 0 and counts["p2"] > 0, (
        f"children only ever came from one parent: {counts}")


def test_child_does_not_alias_its_parents_spec(exp):
    """
    A shared spec object would let a later mutation of the child rewrite a parent
    that is still in the population.
    """
    ss = _space(exp, quantum_amplitude_qubits_range=(4, 10),
                quantum_angle_qubits_range=(4, 10))
    p1 = _parent(exp, ss, 0, encoding="amplitude", n_qubits=5)
    p2 = _parent(exp, ss, 1, encoding="amplitude", n_qubits=5)
    before1, before2 = _sig(p1), _sig(p2)

    seed_everything(7)
    child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
    assert child.quantum_block is not p1.quantum_block
    assert child.quantum_block is not p2.quantum_block

    child.quantum_block.n_qubits = 9
    child.quantum_block.encoding = "angle"
    assert _sig(p1) == before1, "mutating the child rewrote parent1"
    assert _sig(p2) == before2, "mutating the child rewrote parent2"


def test_identical_parents_breed_true(exp):
    """If both parents agree on a circuit, every child must carry it."""
    ss = _space(exp, quantum_amplitude_qubits_range=(4, 12),
                quantum_angle_qubits_range=(4, 12))
    p1 = _parent(exp, ss, 0, encoding="angle", readout="prob",
                 n_qubits=7, reupload=True)
    p2 = _parent(exp, ss, 1, encoding="angle", readout="prob",
                 n_qubits=7, reupload=True)
    want = _sig(p1)
    for i in range(20):
        seed_everything(200 + i)
        child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
        assert _sig(child) == want, f"child {i} drifted to {_sig(child)}"


def test_inherited_genes_still_build(exp):
    """Inheritance must not smuggle in a combination that cannot construct."""
    import torch
    from nas_ts.models.model_builder_v2 import build_model

    ss = _space(exp, quantum_amplitude_qubits_range=(4, 8),
                quantum_angle_qubits_range=(4, 8))
    p1 = _parent(exp, ss, 0, encoding="amplitude", readout="state", n_qubits=6)
    p2 = _parent(exp, ss, 1, encoding="angle", readout="expval_z", n_qubits=5)
    for i in range(6):
        seed_everything(300 + i)
        child = crossover_genome(p1, p2, ss, exp.evolution, exp.genome_constraints)
        model = build_model(child, exp.eval_config.task, d_in=4, d_out=4, num_classes=3)
        with torch.no_grad():
            assert model(torch.randn(2, 1, 4)).shape == (2, 3)
