"""
The four circuit genes must live on the genome, not just in the model builder.

model_builder_v2 has always read encoding / readout / n_qubits / reupload off
QuantumBlockSpec via getattr defaults, so before these fields existed the
defaults silently won and the search could not reach any other circuit. Adding
them to the spec is what makes them searchable at all; these tests pin that they
survive serialisation and appear in a dumped genome, because a gene that does not
round-trip is a gene that is lost every time a genome crosses a process boundary.
"""

import dataclasses

import pytest

from nas_ts.core.genome_v2 import Genome, QuantumBlockSpec

GENES = ("encoding", "readout", "n_qubits", "reupload")


def _genome(**q):
    g = Genome(family="iT", model_dim=64, num_heads=8, ff_mult=2.0,
               pos_encoding="none", dropout=0.0)
    for k, v in q.items():
        setattr(g.quantum_block, k, v)
    return g


def test_spec_has_all_four_genes():
    fields = {f.name for f in dataclasses.fields(QuantumBlockSpec)}
    missing = [g for g in GENES if g not in fields]
    assert not missing, f"QuantumBlockSpec is missing {missing}"


def test_defaults_match_the_previous_builder_behaviour():
    """
    These are the exact getattr defaults in model_builder_v2, so an untouched
    genome must build the block it built before the genes existed.
    """
    q = QuantumBlockSpec()
    assert q.encoding == "amplitude"
    assert q.readout == "state"
    assert q.n_qubits == 0
    assert q.reupload is False


def test_to_dict_carries_the_genes():
    d = _genome(encoding="angle", readout="expval_z", n_qubits=6, reupload=True).to_dict()
    q = d["quantum_block"]
    assert q["encoding"] == "angle"
    assert q["readout"] == "expval_z"
    assert q["n_qubits"] == 6
    assert q["reupload"] is True


@pytest.mark.parametrize("values", [
    {"encoding": "angle", "readout": "expval_z", "n_qubits": 6, "reupload": True},
    {"encoding": "amplitude", "readout": "prob", "n_qubits": 12, "reupload": False},
    {"encoding": "amplitude", "readout": "state", "n_qubits": 0, "reupload": False},
])
def test_round_trip_preserves_every_gene(values):
    before = _genome(enabled=True, **values)
    after = Genome.from_dict(before.to_dict())
    for gene, want in values.items():
        assert getattr(after.quantum_block, gene) == want, f"{gene} lost in round-trip"
    assert after.quantum_block.enabled is True


def test_round_trip_is_stable_across_two_hops():
    """A genome crosses process boundaries more than once in a search."""
    g1 = _genome(enabled=True, encoding="angle", readout="prob", n_qubits=9, reupload=True)
    g2 = Genome.from_dict(g1.to_dict())
    g3 = Genome.from_dict(g2.to_dict())
    assert g3.to_dict()["quantum_block"] == g1.to_dict()["quantum_block"]


def test_old_checkpoints_without_the_genes_still_load():
    """A genome dict written before the genes existed must load, with defaults."""
    d = _genome().to_dict()
    for gene in GENES:
        d["quantum_block"].pop(gene)
    back = Genome.from_dict(d)
    assert back.quantum_block.encoding == "amplitude"
    assert back.quantum_block.readout == "state"
    assert back.quantum_block.n_qubits == 0
    assert back.quantum_block.reupload is False


def test_dumped_genome_names_all_four_genes():
    text = _genome(enabled=True, encoding="angle", readout="expval_z",
                   n_qubits=7, reupload=True).to_structure_str()
    for gene in GENES:
        assert f"{gene}=" in text, f"to_structure_str does not show {gene}"
    assert "encoding=angle" in text
    assert "readout=expval_z" in text
    assert "n_qubits=7" in text
    assert "reupload=True" in text
