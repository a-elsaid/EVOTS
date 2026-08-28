"""Serializable description of a trained model.

Worker processes cannot hand a live ``nn.Module`` back to the main process:
pickle chokes on non-reconstructible attributes (a compiled tensorcircuit
closure, for one), and a large model would push hundreds of megabytes through
the pool's result channel. A ``ModelPackage`` carries only what is genuinely
needed to reconstitute the model on the other side -- a genome description, a
flat CPU ``state_dict``, dataset ``meta``, and the metrics -- all of which are
plainly picklable.

Typical use:

    # worker
    pkg = ModelPackage.from_trained(genome, model, meta, metrics)
    path = pkg.save(run_dir / f"pkg_{indiv_id}.pt")
    return path                      # not the tensors

    # main
    pkg = ModelPackage.load(path)
    model = pkg.rebuild(task)        # strict load; raises on any mismatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..core.genome_v2 import Genome


@dataclass
class ModelPackage:
    """Everything needed to rebuild a trained model in another process.

    Carries a ``state_dict``, never a live ``nn.Module`` -- that is what makes it
    survive the process boundary.
    """

    genome_dict: Dict[str, Any]                      # from genome.to_dict()
    state_dict: Dict[str, torch.Tensor]              # flat CPU tensors
    meta: Dict[str, Any] = field(default_factory=dict)      # dataset scaling/shape info
    metrics: Dict[str, Any] = field(default_factory=dict)   # fitness, val mse, eval id

    # ------------------------------------------------------------------
    @classmethod
    def from_trained(
        cls,
        genome: Genome,
        model: nn.Module,
        meta: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> "ModelPackage":
        """Snapshot a trained model. Tensors are detached and moved to CPU."""
        return cls(
            genome_dict=genome.to_dict(),
            state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
            meta=dict(meta or {}),
            metrics=dict(metrics or {}),
        )

    @classmethod
    def from_state_dict(
        cls,
        genome: Genome,
        state_dict: Dict[str, torch.Tensor],
        meta: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> "ModelPackage":
        """Build from an already-extracted state_dict (what evaluate() returns)."""
        return cls(
            genome_dict=genome.to_dict(),
            state_dict={k: v.detach().cpu() for k, v in (state_dict or {}).items()},
            meta=dict(meta or {}),
            metrics=dict(metrics or {}),
        )

    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> Path:
        """Write to disk and return the path. Parent directories are created."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "genome_dict": self.genome_dict,
                "state_dict": self.state_dict,
                "meta": self.meta,
                "metrics": self.metrics,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelPackage":
        """Read a package written by :meth:`save`."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            genome_dict=blob["genome_dict"],
            state_dict=blob["state_dict"],
            meta=blob.get("meta", {}) or {},
            metrics=blob.get("metrics", {}) or {},
        )

    # ------------------------------------------------------------------
    @property
    def genome(self) -> Genome:
        """Reconstruct the Genome from its dict form."""
        return Genome.from_dict(self.genome_dict)

    def rebuild(self, task) -> nn.Module:
        """Rebuild the model and load the weights strictly.

        ``strict=True`` is deliberate: after the tokenizer projections were moved
        into ``__init__``, any missing or unexpected key means the architecture
        and the weights genuinely disagree, which should fail loudly rather than
        silently leave a layer randomly initialised.
        """
        # Imported here to avoid a circular import at module load time.
        from ..models.model_builder_v2 import build_model_from_meta

        model = build_model_from_meta(self.genome, task, self.meta)
        model.load_state_dict(self.state_dict, strict=True)
        return model
