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

import pickle

import pytest
import torch

import nas_ts.models.model_builder_v2 as M
from nas_ts.models.model_builder_v2 import (
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
    """Guards this file itself: a new key field must gain a case above."""
    covered = {case[0] for case in KEY_FIELD_CASES}
    clear_circuit_cache()
    _compile(_block())
    key = circuit_cache_info()["keys"][0]
    assert len(key) == len(covered), (
        f"cache key has {len(key)} components but only {len(covered)} fields are "
        f"exercised by KEY_FIELD_CASES: {sorted(covered)}"
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
        readouts = [k[5] for k in circuit_cache_info()["keys"]]
        assert "state" not in readouts, "LRU should have evicted the oldest entry"

        torch.manual_seed(5)
        again = _compile(_block(readout="state"))
        assert torch.equal(reference, again), "recompiled circuit changed the output"
    finally:
        M._CIRCUIT_CACHE_MAXSIZE = original_max
        clear_circuit_cache()


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
