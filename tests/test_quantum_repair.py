"""
repair_genome must leave every quantum block buildable.

The failure this guards is silent: angle encoding raises at build time when
n_qubits is 0, the worker catches it and scores the genome inf, and an inf score
is indistinguishable from a genuinely bad architecture. A whole search can
therefore reject every angle genome without anything in the logs saying so.

These tests build the models rather than inspecting fields, because "in range"
is not the property that matters -- "constructs" is.
"""

import dataclasses
import random

import pytest
import torch

from nas_ts.core.config import SearchSpaceConfig
from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.models.model_builder_v2 import build_model
from nas_ts.search.genome_init import random_genome
from nas_ts.search.repair import _qubits_range, repair_genome
from nas_ts.utils.seeding import seed_everything

CONFIG = "configs/classification/iris.yml"
D_IN, N_CLASSES = 4, 3


@pytest.fixture(scope="module")
def exp():
    return build_experiment(load_cfg(CONFIG, overrides=None))


def _space(exp, **overrides):
    """A copy of the config's search space with quantum ranges pinned."""
    return dataclasses.replace(exp.search_space, **overrides)


def _quantum_genome(exp, ss, seed, all_blocks=True):
    seed_everything(seed)
    g = random_genome(ss, exp.genome_constraints)
    for st in g.stages:
        for b in (st.blocks if all_blocks else st.blocks[:1]):
            b.block_type = "quantum"
    return repair_genome(g, ss)


def _builds(g, exp):
    model = build_model(g, exp.eval_config.task, d_in=D_IN, d_out=D_IN,
                        num_classes=N_CLASSES)
    with torch.no_grad():
        out = model(torch.randn(2, 1, D_IN))
    assert out.shape == (2, N_CLASSES)
    return model


# ------------------------------------------------------- the load-bearing test

@pytest.mark.parametrize("encoding", ["amplitude", "angle"])
def test_repaired_genomes_always_build(exp, encoding):
    """Many random genomes, one encoding pinned: every one must construct."""
    ss = _space(exp, quantum_encodings=[encoding],
                quantum_amplitude_qubits_range=(4, 10),
                quantum_angle_qubits_range=(4, 10))
    for seed in range(12):
        g = _quantum_genome(exp, ss, seed)
        assert g.quantum_block.encoding == encoding
        if encoding == "angle":
            assert g.quantum_block.n_qubits > 0, (
                f"seed {seed}: angle left n_qubits=0, which raises at build time")
        _builds(g, exp)


def test_repaired_genomes_build_with_both_encodings_in_play(exp):
    ss = _space(exp, quantum_amplitude_qubits_range=(4, 9),
                quantum_angle_qubits_range=(4, 9))
    seen = set()
    for seed in range(16):
        g = _quantum_genome(exp, ss, seed)
        seen.add(g.quantum_block.encoding)
        _builds(g, exp)
    assert seen == {"amplitude", "angle"}, f"only saw {seen}; the other path is untested"


# ------------------------------------------------------ newly-enabled branch

def test_newly_enabled_block_gets_every_gene_in_range(exp):
    ss = _space(exp, quantum_encodings=["angle"], quantum_readouts=["expval_z"],
                quantum_angle_qubits_range=(5, 7), quantum_reupload_options=[True])
    for seed in range(8):
        g = _quantum_genome(exp, ss, seed)
        q = g.quantum_block
        assert q.encoding == "angle"
        assert q.readout == "expval_z"
        assert q.reupload is True
        assert 5 <= q.n_qubits <= 7


def test_newly_enabled_never_leaves_angle_at_zero(exp):
    """The specific inf-fitness trap, asserted directly."""
    ss = _space(exp, quantum_encodings=["angle"], quantum_angle_qubits_range=(4, 6))
    for seed in range(20):
        g = _quantum_genome(exp, ss, seed)
        assert g.quantum_block.n_qubits >= 4


# --------------------------------------------------- clamp-if-out-of-range

def test_out_of_range_values_are_pulled_back(exp):
    ss = _space(exp, quantum_encodings=["angle"], quantum_readouts=["state"],
                quantum_angle_qubits_range=(4, 6), quantum_reupload_options=[False])
    g = _quantum_genome(exp, ss, 0)
    # a genome that already had a quantum block, carrying junk
    g.quantum_block.encoding = "nonsense"
    g.quantum_block.readout = "nonsense"
    g.quantum_block.n_qubits = 99
    g.quantum_block.reupload = "nonsense"
    g = repair_genome(g, ss)

    assert g.quantum_block.encoding == "angle"
    assert g.quantum_block.readout == "state"
    assert 4 <= g.quantum_block.n_qubits <= 6
    assert g.quantum_block.reupload is False
    _builds(g, exp)


def test_switching_encoding_redraws_the_width_against_the_new_range(exp):
    """
    amplitude and angle have separate ranges. A genome that mutates from one to
    the other must not keep a width that is only legal for the old encoding.
    """
    ss = _space(exp, quantum_encodings=["angle"],
                quantum_amplitude_qubits_range=(11, 12),
                quantum_angle_qubits_range=(4, 5))
    g = _quantum_genome(exp, ss, 0)
    g.quantum_block.encoding = "amplitude"
    g.quantum_block.n_qubits = 12          # legal for amplitude, not for angle
    g.quantum_block.encoding = "angle"     # ... then the encoding flips
    g = repair_genome(g, ss)
    assert 4 <= g.quantum_block.n_qubits <= 5, (
        f"kept n_qubits={g.quantum_block.n_qubits} from the amplitude range")
    _builds(g, exp)


def test_in_range_values_are_left_alone(exp):
    """Repair fixes what is broken; it must not resample a valid genome."""
    ss = _space(exp, quantum_encodings=["amplitude", "angle"],
                quantum_amplitude_qubits_range=(4, 10),
                quantum_angle_qubits_range=(4, 10))
    g = _quantum_genome(exp, ss, 3)
    before = dataclasses.asdict(g.quantum_block)
    for _ in range(5):
        g = repair_genome(g, ss)
    assert dataclasses.asdict(g.quantum_block) == before, "repair is not idempotent"


# ------------------------------------------- amplitude's native-width path

def test_amplitude_zero_is_preserved_as_the_native_width(exp):
    """
    0 means n = log2(d_model) with no input projection -- the cheapest amplitude
    configuration and what every pre-gene genome had. Repair must not overwrite it.
    """
    ss = _space(exp, quantum_encodings=["amplitude"],
                quantum_amplitude_qubits_range=(4, 10))
    g = _quantum_genome(exp, ss, 1)
    g.quantum_block.n_qubits = 0
    g = repair_genome(g, ss)
    assert g.quantum_block.n_qubits == 0
    model = _builds(g, exp)
    qb = [m for m in model.modules() if type(m).__name__ == "QuantumMixBlock"][0]
    assert qb.in_proj is None, "native width should need no input projection"
    assert 2 ** qb.circuit_spec.n_qubits == g.model_dim


def test_amplitude_zero_is_replaced_when_d_model_is_not_a_power_of_two(exp):
    """The safety net: 0 is only legal while d_model is a power of two."""
    ss = _space(exp, quantum_encodings=["amplitude"],
                quantum_amplitude_qubits_range=(4, 6))
    g = _quantum_genome(exp, ss, 2)
    g.quantum_block.n_qubits = 0
    g.model_dim = 40                       # not a power of two
    from nas_ts.search.repair import _repair_quantum_qubits
    _repair_quantum_qubits(g, ss)
    assert 4 <= g.quantum_block.n_qubits <= 6


# ------------------------------------------------------------------ helper

def test_qubits_range_picks_the_encoding_specific_range():
    ss = SearchSpaceConfig(
        families=["iT"], depth_range=(2, 2), model_dim_range=(32, 64),
        num_heads_range=(4, 4), ff_mult_range=(1.0, 1.0), patch_sizes=[4],
        strides=[1], overlaps=[0.0], per_channel_options=[True], cross_groups=[1],
        cross_fusions=["add"], freq_types=["none"], freq_keep_ratios=[0.5],
        decomp_modes=["none"], decomp_kernel_sizes=[3], conv_kernel_sizes=[3],
        conv_dilations=[1], block_types=["attn"], pos_encoding_options=["none"],
        dropout_range=(0.0, 0.1),
        quantum_amplitude_qubits_range=(4, 12),
        quantum_angle_qubits_range=(6, 14),
    )
    assert _qubits_range(ss, "amplitude") == (4, 12)
    assert _qubits_range(ss, "angle") == (6, 14)
    assert _qubits_range(ss, "anything-else") == (6, 14)   # non-amplitude -> angle
