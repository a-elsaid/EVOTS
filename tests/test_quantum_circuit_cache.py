"""Tests for the process-wide compiled-circuit cache in QuantumMixBlock.

Compiling one circuit costs ~5 s, and each new input shape retraces at ~0.6 s. Every
block instance used to pay both, roughly 8 s per genome evaluation. The cache shares
the compiled function across blocks with an identical configuration.

The dangerous failure mode is UNDER-keying: if two configurations that produce
different circuits map to the same cache key, the second block silently receives the
first's circuit and trains a model that is not the one its genome describes. Nothing
raises. The bulk of this file is therefore one case per key field, asserting that
changing that field alone produces a separate cache entry.

Over-keying is the safe direction and is not tested against: `gate_set` only matters
via `gpl = 3 if gate_set == "rx_ry_rz" else 2`, so two distinct non-"rx_ry_rz" values
key separately but compile identically. That wastes a slot; it never returns a wrong
circuit.
"""

from __future__ import annotations

import inspect
import pickle
from dataclasses import FrozenInstanceError, fields

import pytest
import torch

import nas_ts.models.model_builder_v2 as M
from nas_ts.models.model_builder_v2 import (
    CircuitSpec,
    QuantumMixBlock,
    circuit_cache_info,
    clear_circuit_cache,
)


D_MODEL = 16          # n = log2(16) = 4 on the default amplitude path
BASE_QUBITS = 4
BATCH, TOKENS = 2, 3


def _block(**overrides) -> QuantumMixBlock:
    kwargs = dict(
        d_model=D_MODEL,
        nlayers=1,                 # keep compiles cheap
        entangle_pattern="linear",
        gate_set="rx_ry",
        use_ffn=False,
        ff_mult=2.0,
        dropout=0.0,
        encoding="amplitude",
        readout="state",
        n_qubits=BASE_QUBITS,
        reupload=False,
    )
    kwargs.update(overrides)
    return QuantumMixBlock(**kwargs)


def _compile(block: QuantumMixBlock) -> torch.Tensor:
    """Force compilation — building a block does not compile, the first forward does."""
    block.eval()
    with torch.no_grad():
        return block(torch.randn(BATCH, TOKENS, D_MODEL))


# Each entry: (field, value_a, value_b, extra kwargs applied to both blocks).
# Every field in the cache key must appear here.
KEY_FIELD_CASES = [
    ("n_qubits", 3, 4, {}),
    ("nlayers", 1, 2, {}),
    ("entangle_pattern", "linear", "circular", {}),
    ("gate_set", "rx_ry", "rx_ry_rz", {}),
    ("encoding", "amplitude", "angle", {}),
    ("readout", "state", "prob", {}),
    # reupload only changes the circuit under angle encoding
    ("reupload", False, True, {"encoding": "angle"}),
]


@pytest.mark.parametrize("field,a,b,extra", KEY_FIELD_CASES,
                         ids=[c[0] for c in KEY_FIELD_CASES])
def test_each_key_field_produces_a_separate_cache_entry(field, a, b, extra):
    """Two blocks differing in exactly one key field must not share a circuit."""
    clear_circuit_cache()

    block_a = _block(**{field: a}, **extra)
    _compile(block_a)
    assert circuit_cache_info()["size"] == 1

    block_b = _block(**{field: b}, **extra)
    _compile(block_b)

    size = circuit_cache_info()["size"]
    assert size == 2, (
        f"changing {field!r} from {a!r} to {b!r} did not create a new cache entry "
        f"(size={size}). The second block is silently reusing the first block's "
        f"compiled circuit, so it computes a different function than its genome describes."
    )
    assert block_a._qpred_torch is not block_b._qpred_torch


def test_identical_config_shares_one_entry():
    """The converse: without this, the test above would pass even if keying were broken."""
    clear_circuit_cache()

    a, b = _block(), _block()
    _compile(a)
    _compile(b)

    assert circuit_cache_info()["size"] == 1
    assert a._qpred_torch is b._qpred_torch


def test_all_key_fields_are_covered_by_the_cases():
    """Guards this file itself: a new field on CircuitSpec must gain a case above.

    CircuitSpec is the single declared source for the circuit's identity, so
    comparing against its field names catches a new gene that was added to the
    circuit but never exercised for cache separation here.
    """
    covered = {case[0] for case in KEY_FIELD_CASES}
    declared = {f.name for f in fields(CircuitSpec)}
    assert declared == covered, (
        f"CircuitSpec fields not exercised by KEY_FIELD_CASES: {sorted(declared - covered)}; "
        f"cases naming fields that no longer exist: {sorted(covered - declared)}"
    )


def test_lru_evicts_oldest_and_recompiles_identically():
    """Eviction is a time/memory trade, never a correctness one."""
    clear_circuit_cache()
    original_max = M._CIRCUIT_CACHE_MAXSIZE
    M._CIRCUIT_CACHE_MAXSIZE = 2
    try:
        torch.manual_seed(5)
        first = _block(readout="state")
        reference = _compile(first)

        _compile(_block(readout="prob"))
        _compile(_block(readout="expval_z"))

        assert circuit_cache_info()["size"] == 2
        readouts = [k.readout for k in circuit_cache_info()["keys"]]
        assert "state" not in readouts, "LRU should have evicted the oldest entry"

        torch.manual_seed(5)
        again = _compile(_block(readout="state"))
        assert torch.equal(reference, again), "recompiled circuit changed the output"
    finally:
        M._CIRCUIT_CACHE_MAXSIZE = original_max
        clear_circuit_cache()


def test_compile_takes_only_the_spec():
    """The structural guarantee behind keying on CircuitSpec.

    The cache key is the spec object, so anything the compiler branches on must
    arrive through the spec. A second parameter would be a value that influences the
    circuit but is absent from the key — the exact silent-collision failure this
    design exists to prevent.
    """
    params = list(inspect.signature(M._compile_qpred).parameters)
    assert params == ["spec"], (
        f"_compile_qpred takes {params}; anything not on CircuitSpec is invisible to "
        "the cache key and will collide silently"
    )


def test_equal_specs_share_an_entry_and_unequal_ones_do_not():
    """The key is the spec's value, not its identity."""
    clear_circuit_cache()
    a, b = _block(), _block()
    assert a.circuit_spec == b.circuit_spec
    assert a.circuit_spec is not b.circuit_spec       # distinct objects
    assert hash(a.circuit_spec) == hash(b.circuit_spec)
    _compile(a)
    _compile(b)
    assert circuit_cache_info()["size"] == 1
    assert circuit_cache_info()["keys"][0] == a.circuit_spec


def test_spec_is_frozen():
    """A mutable spec could describe a different circuit than the one cached under it."""
    spec = _block().circuit_spec
    with pytest.raises(FrozenInstanceError):
        spec.readout = "prob"


def test_block_does_not_duplicate_circuit_fields():
    """Circuit config is read through to the spec, so the two cannot disagree."""
    block = _block()
    for name in (f.name for f in fields(CircuitSpec)):
        assert name not in block.__dict__, (
            f"{name!r} is stored on the block as well as on the spec; a copy can drift "
            "out of sync with the value the cache is keyed on"
        )
    # the read-through accessors still work
    assert block.n_qubits == block.circuit_spec.n_qubits
    assert block.encoding == block.circuit_spec.encoding
    assert block.readout == block.circuit_spec.readout
    assert block.readout_width == block.circuit_spec.readout_width


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"readout": "counts"}, "readout"),
        ({"gate_set": "rx"}, "gate_set"),
        ({"entangle_pattern": "ring"}, "entangle_pattern"),
        ({"encoding": "basis"}, "encoding"),
        ({"n_qubits": 0}, "n_qubits"),
        ({"nlayers": 0}, "nlayers"),
    ],
)
def test_spec_validates_itself(overrides, expected):
    """CircuitSpec is independently constructible, so it owns its invariants."""
    kwargs = dict(n_qubits=4, nlayers=1, entangle_pattern="linear", gate_set="rx_ry",
                  encoding="amplitude", readout="state", reupload=False)
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=expected):
        CircuitSpec(**kwargs)


def test_spec_canonicalises_types():
    """A bool-valued int and a real bool must not occupy two cache slots."""
    a = CircuitSpec(n_qubits=4, nlayers=1, entangle_pattern="linear", gate_set="rx_ry",
                    encoding="amplitude", readout="state", reupload=0)
    b = CircuitSpec(n_qubits=4, nlayers=1, entangle_pattern="linear", gate_set="rx_ry",
                    encoding="amplitude", readout="state", reupload=False)
    assert a == b and hash(a) == hash(b)
    assert a.reupload is False


def test_cache_is_not_reachable_from_instance_state():
    """The cached value is an unpicklable closure; it must not escape into __dict__.

    __getstate__ drops _qpred_torch for exactly this reason. If the cache container
    itself became instance state, that protection would be bypassed.
    """
    clear_circuit_cache()
    block = _block()
    _compile(block)

    assert not any(v is M._CIRCUIT_CACHE for v in block.__dict__.values())
    assert "_CIRCUIT_CACHE" not in block.__dict__
    assert block.__getstate__()["_qpred_torch"] is None

    restored = pickle.loads(pickle.dumps(block))
    with torch.no_grad():
        out = restored(torch.randn(BATCH, TOKENS, D_MODEL))
    assert out.shape == (BATCH, TOKENS, D_MODEL)
