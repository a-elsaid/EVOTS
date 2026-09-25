"""QuantumMixBlock rejects unusable configurations loudly instead of falling through.

Every branch in the circuit builder selects behaviour by string comparison:

    gpl     = 3 if gate_set == "rx_ry_rz" else 2
    circular = pattern == "circular"

so any unrecognised value silently degrades to rx_ry / linear. The block would build,
train and report a fitness for a circuit that is not the one its genome describes —
the same defect class as the orphaned stages and the finetune restore bug, both of
which were silent for exactly this reason.

The search itself cannot produce these values (repair range-checks them against the
search space), but a hand-edited genome JSON or a widened config can, so the block
validates rather than trusting its caller.
"""

from __future__ import annotations

import pytest

from nas_ts.core.config import SearchSpaceConfig
from nas_ts.models.model_builder_v2 import (
    QUANTUM_ENCODINGS,
    QUANTUM_ENTANGLE_PATTERNS,
    QUANTUM_GATE_SETS,
    QUANTUM_READOUTS,
    QuantumMixBlock,
)


D_MODEL = 16  # n = log2(16) = 4


def _block(**overrides) -> QuantumMixBlock:
    kwargs = dict(
        d_model=D_MODEL,
        nlayers=1,
        entangle_pattern="linear",
        gate_set="rx_ry",
        use_ffn=False,
        ff_mult=2.0,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return QuantumMixBlock(**kwargs)


# (description, overrides, substring expected in the message)
REJECTED = [
    ("unknown gate_set", {"gate_set": "rx_ry_rzz"}, "gate_set"),
    ("unknown entangle_pattern", {"entangle_pattern": "ring"}, "entangle_pattern"),
    ("unknown encoding", {"encoding": "basis"}, "encoding"),
    ("unknown readout", {"readout": "counts"}, "readout"),
    ("angle encoding without n_qubits", {"encoding": "angle", "n_qubits": 0}, "n_qubits"),
    ("non-power-of-2 d_model, derived n", {"d_model": 24}, "power of 2"),
]


@pytest.mark.parametrize("desc,overrides,expected", REJECTED,
                         ids=[c[0] for c in REJECTED])
def test_invalid_configuration_raises(desc, overrides, expected):
    with pytest.raises(ValueError, match=expected):
        _block(**overrides)


@pytest.mark.parametrize("gate_set", QUANTUM_GATE_SETS)
def test_every_valid_gate_set_is_accepted(gate_set):
    """The guard must not be so tight it rejects values the search will generate."""
    assert _block(gate_set=gate_set) is not None


@pytest.mark.parametrize("pattern", QUANTUM_ENTANGLE_PATTERNS)
def test_every_valid_entangle_pattern_is_accepted(pattern):
    assert _block(entangle_pattern=pattern) is not None


@pytest.mark.parametrize(
    "field,accepted",
    [
        ("quantum_gate_sets", QUANTUM_GATE_SETS),
        ("quantum_entangle_patterns", QUANTUM_ENTANGLE_PATTERNS),
        ("quantum_encodings", QUANTUM_ENCODINGS),
        ("quantum_readouts", QUANTUM_READOUTS),
    ],
)
def test_builder_accepts_everything_the_search_space_can_generate(field, accepted):
    """Catch drift between what the search can emit and what the block will accept.

    If the search space is widened without updating the builder's tuple, every genome
    using the new value raises at build time and scores inf — which in the logs is
    indistinguishable from a legitimately bad architecture.
    """
    searchable = SearchSpaceConfig.__dataclass_fields__[field].default_factory()
    unbuildable = sorted(set(searchable) - set(accepted))
    assert not unbuildable, (
        f"SearchSpaceConfig.{field} can generate {unbuildable}, which QuantumMixBlock "
        f"rejects (it accepts {list(accepted)})"
    )
