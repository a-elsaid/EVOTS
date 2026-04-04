#!/usr/bin/env python3
from typing import Dict, Optional, List
import copy
import torch
import hashlib

from ..core.genome_v2 import Genome, BlockSpec


def _block_signature(b: BlockSpec) -> str:
    # We keep only architecture-defining fields (no norms for now).
    parts = [
        b.block_type,
        str(b.dim) if b.dim is not None else "None",
        str(b.num_heads) if b.num_heads is not None else "None",
        # f"{b.ff_mult:.3f}" if b.ff_mult is not None else "None",
    ]
    return ":".join(parts)

def architecture_signature(genome: Genome) -> str:
    """
    Returns a short hash string representing the architecture.
    This is used as a key for the weight pool.
    """
    parts: List[str] = []

    parts.append(genome.family)
    parts.append(str(genome.model_dim))
    parts.append(str(genome.num_heads))
    parts.append(f"{genome.ff_mult:.3f}")

    # Backbone blocks
    if not genome.stages:
        raise ValueError(f"Genome {genome.id} has no stages — cannot compute architecture signature")
    for st in (genome.stages or []):
        block_sigs = ",".join(_block_signature(b) for b in (st.blocks or []))
        parts.append(f"stage:{st.tokenizer}:{st.retokenize}:[{block_sigs}]")

    s = "|".join(parts)
    # Hash to keep the key short and uniform
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h


class WeightPool:
    """
    Simple in-memory weight pool keyed by architecture signature.
    It stores *state_dicts* of trained models.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._pool: Dict[str, Dict[str, torch.Tensor]] = {}

    def to_config_dict(self) -> dict:
        # NOTE: NO  weights serialization -- only pool is enabled/disabled
        return {"enabled": self.enabled}

    def get(self, genome: Genome) -> Optional[Dict[str, torch.Tensor]]:
        key = architecture_signature(genome)
        state = self._pool.get(key, None)
        return copy.deepcopy(state) if state is not None else None

    def update(self, genome: Genome, state_dict: Dict[str, torch.Tensor]):
        key = architecture_signature(genome)
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        self._pool[key] = cpu_state