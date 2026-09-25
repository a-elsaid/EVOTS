"""
The quantum-condition configs must be runnable unattended, like the classical ones.

Same contract as tests/test_classification_configs.py, plus the quantum-specific
hazards: a genome whose circuit cannot build scores inf and reads as a bad
architecture, and a qubit range set too wide makes workers OOM, which surfaces
the same way. So these tests sample the space and actually construct models,
including after repeated mutation.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.models.model_builder_v2 import (
    QUANTUM_ENCODINGS, QUANTUM_READOUTS, build_model,
)
from nas_ts.search.genome_init import random_genome
from nas_ts.search.mutation import mutate_genome
from nas_ts.search.repair import repair_genome
from nas_ts.utils.seeding import seed_everything

QUANTUM_DIR = Path("configs/classification_quantum")
CLASSICAL_DIR = Path("configs/classification")

EXPECTED = {
    "iris":          dict(d_in=4,  k=3, angle_hi=14),
    "wine":          dict(d_in=13, k=3, angle_hi=14),
    "seeds":         dict(d_in=7,  k=3, angle_hi=14),
    "breast_cancer": dict(d_in=30, k=2, angle_hi=12),
}


def _cfg(name):
    return load_cfg(str(QUANTUM_DIR / f"{name}.yml"), overrides=None)


# ------------------------------------------------------- the suite's contract

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_config_exists_under_the_expected_filename(name):
    """--config-dir looks up <dataset>.yml, so the names must match exactly."""
    assert (QUANTUM_DIR / f"{name}.yml").exists()
    build_experiment(_cfg(name))


def test_directory_holds_exactly_the_four_datasets():
    found = sorted(p.stem for p in QUANTUM_DIR.glob("*.yml"))
    assert found == sorted(EXPECTED), f"unexpected contents: {found}"


# ------------------------------------------------ same budget as classical

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_budget_and_task_match_the_classical_config(name):
    """
    The two conditions are only comparable if nothing but the block space differs.
    """
    q = _cfg(name)
    c = load_cfg(str(CLASSICAL_DIR / f"{name}.yml"), overrides=None)

    assert q["task"] == c["task"]
    assert q["data"] == c["data"]
    assert q["eval"] == c["eval"]
    assert q["constraints"] == c["constraints"]
    assert q["selection"] == c["selection"]
    assert {k: v for k, v in q["evo"].items()} == {k: v for k, v in c["evo"].items()}

    # ... and the search space differs only in block_types plus the new genes.
    q_ss, c_ss = dict(q["search_space"]), dict(c["search_space"])
    assert q_ss.pop("block_types") == c_ss.pop("block_types") + ["quantum"]
    for key in list(q_ss):
        if key.startswith("quantum_"):
            q_ss.pop(key)
    assert q_ss == c_ss, "search spaces differ beyond the quantum additions"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_quantum_is_in_the_block_space(name):
    assert "quantum" in _cfg(name)["search_space"]["block_types"]


# ------------------------------------------------------------ gene ranges

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_gene_ranges_are_the_measured_ones(name):
    ss = _cfg(name)["search_space"]
    assert ss["quantum_amplitude_qubits_range"] == [4, 12]
    assert ss["quantum_angle_qubits_range"] == [4, EXPECTED[name]["angle_hi"]]
    assert sorted(ss["quantum_encodings"]) == sorted(QUANTUM_ENCODINGS)
    assert sorted(ss["quantum_readouts"]) == sorted(QUANTUM_READOUTS)


def test_breast_cancer_angle_range_is_capped_below_the_others():
    """
    It vmaps 960 tokens per call against iris's 64; the angle/expval_z corner at
    n=14 measured 7.7 GB there, ~85 GB across 11 workers.
    """
    bc = _cfg("breast_cancer")["search_space"]["quantum_angle_qubits_range"]
    iris = _cfg("iris")["search_space"]["quantum_angle_qubits_range"]
    assert bc[1] < iris[1]


# ------------------------------------------- the load-bearing build tests

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_sampled_genomes_build_and_run(name):
    cfg = _cfg(name)
    exp = build_experiment(cfg)
    d_in, k = EXPECTED[name]["d_in"], EXPECTED[name]["k"]

    seed_everything(0)
    saw_quantum = 0
    for i in range(25):
        g = random_genome(exp.search_space, exp.genome_constraints)
        model = build_model(g, exp.eval_config.task, d_in=d_in, d_out=d_in,
                            num_classes=k).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, d_in))
        assert out.shape == (2, k), f"genome {i} gave {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"genome {i} produced non-finite logits"
        if g.quantum_block.enabled:
            saw_quantum += 1
    assert saw_quantum > 0, "no sampled genome contained a quantum block"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_mutated_genomes_stay_buildable(name):
    """The search mutates for 500 genomes; repair must keep every one valid."""
    cfg = _cfg(name)
    exp = build_experiment(cfg)
    d_in, k = EXPECTED[name]["d_in"], EXPECTED[name]["k"]

    seed_everything(1)
    g = random_genome(exp.search_space, exp.genome_constraints)
    for i in range(20):
        g = mutate_genome(g, exp.search_space, exp.evolution, exp.genome_constraints)
        model = build_model(g, exp.eval_config.task, d_in=d_in, d_out=d_in,
                            num_classes=k).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, d_in))
        assert out.shape == (2, k), f"after mutation {i}: {tuple(out.shape)}"


def test_every_encoding_and_readout_is_reachable_and_builds():
    """
    Pin each circuit combination in turn and build it, so no corner of the space
    is left untested by sampling luck.
    """
    exp = build_experiment(_cfg("iris"))
    seed_everything(2)
    for encoding in QUANTUM_ENCODINGS:
        for readout in QUANTUM_READOUTS:
            g = random_genome(exp.search_space, exp.genome_constraints)
            for st in g.stages:
                for b in st.blocks:
                    b.block_type = "quantum"
            g.quantum_block.enabled = True
            g.quantum_block.encoding = encoding
            g.quantum_block.readout = readout
            g.quantum_block.n_qubits = 5
            g = repair_genome(g, exp.search_space)
            model = build_model(g, exp.eval_config.task, d_in=4, d_out=4,
                                num_classes=3).eval()
            with torch.no_grad():
                out = model(torch.randn(2, 1, 4))
            assert out.shape == (2, 3), f"{encoding}/{readout} gave {tuple(out.shape)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_qubit_widths_stay_inside_the_configured_ranges(name):
    """A width outside the measured range is the OOM risk, so pin it."""
    exp = build_experiment(_cfg(name))
    amp_lo, amp_hi = exp.search_space.quantum_amplitude_qubits_range
    ang_lo, ang_hi = exp.search_space.quantum_angle_qubits_range

    seed_everything(3)
    g = random_genome(exp.search_space, exp.genome_constraints)
    for i in range(40):
        g = mutate_genome(g, exp.search_space, exp.evolution, exp.genome_constraints)
        q = g.quantum_block
        if not q.enabled:
            continue
        if q.encoding == "angle":
            assert ang_lo <= q.n_qubits <= ang_hi, f"angle n={q.n_qubits} outside {(ang_lo, ang_hi)}"
        else:
            assert q.n_qubits == 0 or amp_lo <= q.n_qubits <= amp_hi, \
                f"amplitude n={q.n_qubits} outside {(amp_lo, amp_hi)}"
