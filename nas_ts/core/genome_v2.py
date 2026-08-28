from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Union
from pathlib import Path
import json


def _block_summary(b: BlockSpec) -> Dict[str, Any]:
    return {
        "type": b.block_type,
        "dim": b.dim,
        "num_heads": b.num_heads,
        "ff_mult": b.ff_mult,
        "norm": b.norm,
    }

def _fmt_opt(x: Any) -> str:
    return "-" if x is None else str(x)

def _short_block(b: BlockSpec) -> str:
    # compact single-line block descriptor
    return (
        f"{b.block_type}"
        f"(h={_fmt_opt(b.num_heads)}, ff={_fmt_opt(b.ff_mult)}, dim={_fmt_opt(b.dim)}, norm={b.norm})"
    )

# ----------------------------
# Small leaf specs (same names as before)
# ----------------------------

@dataclass
class BlockSpec:
    block_type: str
    dim: Optional[int] = None
    num_heads: Optional[int] = None
    ff_mult: Optional[float] = None
    norm: str = "layernorm"


@dataclass
class PatchingSpec:
    enabled: bool = False
    patch_size: int = 16
    stride: int = 16
    overlap: float = 0.0
    per_channel: bool = True


# @dataclass
# class CrossDimSpec:
#     enabled: bool = False
#     groups: int = 1
#     fusion: str = "add"


# @dataclass
# class FreqSpec:
#     enabled: bool = False
#     freq_type: str = "fft"
#     keep_ratio: float = 0.5


# @dataclass
# class DecompSpec:
#     enabled: bool = False
#     mode: str = "moving_avg"
#     kernel_size: int = 3


# @dataclass
# class ConvSpec:
#     enabled: bool = False
#     kernel_size: int = 3
#     dilation: int = 1


# ----------------------------
# Tokenizer specs (new but compatible with your earlier idea)
# ----------------------------

@dataclass
class VarTokenHeadSpec:
    enabled: bool = False
    encoder_type: str = "linear"       # "linear" | "conv" | "fft" | "decomp_linear"
    conv_kernel: int = 5
    conv_stride: int = 1
    conv_pool: str = "avg"             # "avg" | "max"
    fft_keep_ratio: float = 0.25
    decomp_kernel: int = 5


@dataclass
class CrossTokenHeadSpec:
    enabled: bool = False
    groups: int = 2
    patch_size: int = 8
    stride: int = 8
    encoder_type: str = "linear"       # "linear" | "conv"
    conv_kernel: int = 3
    pool: str = "avg"                  # "avg" | "max"


@dataclass
class ConvBlockSpec:
    enabled: bool = False
    kernel_size: int = 3    # depthwise conv kernel (odd int)
    dilation: int = 1       # dilation factor; receptive field = dilation*(kernel_size-1)+1


@dataclass
class QuantumBlockSpec:
    enabled: bool = False
    nlayers: int = 2                    # circuit depth (entangle+rotate layers)
    entangle_pattern: str = "linear"    # "linear" | "circular"
    gate_set: str = "rx_ry"            # "rx_ry" | "rx_ry_rz"
    use_ffn: bool = True                # append FFN after the quantum residual


# ----------------------------
# NEW: StageSpec
# ----------------------------

@dataclass
class StageSpec:
    """
    A stage is:
      tokenizer(raw_x) -> tokens
      optional cross-attn re-tokenization using (raw_x + prev_tokens)
      core(tokens) -> tokens
    """
    name: str = "stage0"

    # "time" | "var" | "patch" | "cross"
    tokenizer: str = "time"

    # cross-attention re-tokenization between tokenizer and core blocks.
    # Not searched: repair_genome() sets this from the stage index —
    # stage0 -> "none" (no predecessor), every stage i>=1 -> "cross_attn".
    retokenize: str = "none"   # "none" | "cross_attn" | "conv" | "fft"

    # core blocks used after tokenization (Transformer-style blocks)
    blocks: List[BlockSpec] = field(default_factory=list)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        d = asdict(obj)
        return {k: _to_plain(v) for k, v in d.items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_plain_blockspec(d: Dict[str, Any]) -> BlockSpec:
    return BlockSpec(
        block_type=d.get("block_type", "attn"),
        dim=d.get("dim"),
        num_heads=d.get("num_heads"),
        ff_mult=d.get("ff_mult"),
        norm=d.get("norm", "layernorm"),
    )


def _from_plain_stage(d: Dict[str, Any]) -> StageSpec:
    return StageSpec(
        name=d.get("name", "stage"),
        tokenizer=d.get("tokenizer", "time"),
        retokenize=d.get("retokenize", "none"),
        blocks=[_from_plain_blockspec(b) for b in d.get("blocks", [])],
    )


# ----------------------------
# Genome (same class name, includes stages)
# ----------------------------

@dataclass
class Genome:
    # kept (global NAS params)
    family: str
    model_dim: int
    num_heads: int
    ff_mult: float
    pos_encoding: str
    dropout: float
    _id_counter: ClassVar[int] = 0
    id: int = field(init=False)

    # classic fields (still here)
    # blocks: List[BlockSpec] = field(default_factory=list)
    # patching: PatchingSpec = field(default_factory=PatchingSpec)
    # cross_dim: CrossDimSpec = field(default_factory=CrossDimSpec)
    # freq_block: FreqSpec = field(default_factory=FreqSpec)
    # decomp_block: DecompSpec = field(default_factory=DecompSpec)
    # conv_block: ConvSpec = field(default_factory=ConvSpec)

    # token heads
    var_head: VarTokenHeadSpec = field(default_factory=VarTokenHeadSpec)
    cross_head: CrossTokenHeadSpec = field(default_factory=CrossTokenHeadSpec)
    conv_block: ConvBlockSpec = field(default_factory=ConvBlockSpec)
    quantum_block: QuantumBlockSpec = field(default_factory=QuantumBlockSpec)

    # NEW: stage program (this enables future “stages” cleanly)
    stages: List[StageSpec] = field(default_factory=list)

    def __post_init__(self):
        # Assign incremental ID
        self.id = Genome._id_counter
        Genome._id_counter += 1
        
        

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "ff_mult": self.ff_mult,
            "pos_encoding": self.pos_encoding,
            "dropout": self.dropout,
            "var_head": _to_plain(self.var_head),
            "cross_head": _to_plain(self.cross_head),
            "conv_block": _to_plain(self.conv_block),
            "quantum_block": _to_plain(self.quantum_block),
            "stages": [_to_plain(s) for s in self.stages],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Genome":
        return cls(
            family=d["family"],
            model_dim=int(d["model_dim"]),
            num_heads=int(d["num_heads"]),
            ff_mult=float(d["ff_mult"]),
            pos_encoding=d.get("pos_encoding", "none"),
            dropout=float(d.get("dropout", 0.0)),
            var_head=VarTokenHeadSpec(**d.get("var_head", {"enabled": False})),
            cross_head=CrossTokenHeadSpec(**d.get("cross_head", {"enabled": False})),
            conv_block=ConvBlockSpec(**d.get("conv_block", {"enabled": False})),
            quantum_block=QuantumBlockSpec(**d.get("quantum_block", {"enabled": False})),
            stages=[_from_plain_stage(s) for s in d.get("stages", [])],
            # v1 fields (patching, cross_dim, freq_block, decomp_block, conv_block, blocks)
            # are intentionally ignored — they no longer exist on Genome
        )

    # --------------------------------------------------
    # NEW: architecture / structure dump (human-facing)
    # --------------------------------------------------
    
    def to_structure_dict(self) -> Dict[str, Any]:
        """
        Return a clean, JSON-safe description of the model structure.
        This is for logging, visualization, and debugging
        """
        return {
            "family": self.family,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "ff_mult": self.ff_mult,
            "pos_encoding": self.pos_encoding,
            "dropout": self.dropout,

            "token_heads": {
                "var_head": _to_plain(self.var_head),
                "cross_head": _to_plain(self.cross_head),
                "conv_block": _to_plain(self.conv_block),
                "quantum_block": _to_plain(self.quantum_block),
            },

            "stages": [
                {
                    "name": s.name,
                    "tokenizer": s.tokenizer,
                    "retokenize": s.retokenize,
                    "num_blocks": len(s.blocks),
                    "blocks": [_block_summary(b) for b in s.blocks],
                }
                for s in self.stages
            ],
        }

    def to_structure_json(self, indent: int = 2) -> str:
        """
        Return the architecture as a JSON string.
        """
        return json.dumps(self.to_structure_dict(), indent=indent)

    def dump_structure(self, path: Union[str, Path], indent: int = 2) -> None:
        """
        Write the architecture JSON to disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_structure_json(indent=indent))


    def to_structure_str(self, *, max_blocks_per_stage: int = 12) -> str:
        """
        Return a pretty, human-readable string showing the architecture.

        max_blocks_per_stage:
          - limits how many blocks we print per stage (keeps logs readable)
        """
        lines: list[str] = []

        # header
        lines.append(
            f"Genome(ID:{self.id}): (family={self.family}, D={self.model_dim}, heads={self.num_heads}, "
            f"ff_mult={self.ff_mult:.3g}, pos={self.pos_encoding}, dropout={self.dropout:.3g})"
        )

        # heads
        lines.append("TokenHeads:")
        lines.append(
            f"  - var_head: enabled={self.var_head.enabled}, enc={self.var_head.encoder_type}, "
            f"conv_k={self.var_head.conv_kernel}, pool={self.var_head.conv_pool}, "
            f"fft_keep={self.var_head.fft_keep_ratio:.3g}, decomp_k={self.var_head.decomp_kernel}"
        )
        lines.append(
            f"  - cross_head: enabled={self.cross_head.enabled}, groups={self.cross_head.groups}, "
            f"p={self.cross_head.patch_size}, s={self.cross_head.stride}, enc={self.cross_head.encoder_type}, "
            f"conv_k={self.cross_head.conv_kernel}, pool={self.cross_head.pool}"
        )
        lines.append(
            f"  - conv_block: enabled={self.conv_block.enabled}, "
            f"kernel={self.conv_block.kernel_size}, dilation={self.conv_block.dilation}, "
            f"rf={self.conv_block.dilation * (self.conv_block.kernel_size - 1) + 1}"
        )
        lines.append(
            f"  - quantum_block: enabled={self.quantum_block.enabled}, "
            f"nlayers={self.quantum_block.nlayers}, "
            f"entangle={self.quantum_block.entangle_pattern}, "
            f"gate_set={self.quantum_block.gate_set}, "
            f"use_ffn={self.quantum_block.use_ffn}"
        )

        # stages
        stages = self.stages or []
        lines.append(f"Stages({len(stages)}):")
        if not stages:
            lines.append("  (none)")
            return "\n".join(lines)

        for si, st in enumerate(stages):
            lines.append(
                f"  [{si}] {st.name}: tokenizer={st.tokenizer}  retokenize={st.retokenize}  blocks={len(st.blocks)}"
            )

            if not st.blocks:
                lines.append("      (no blocks)")
                continue

            # print blocks, truncated
            n = len(st.blocks)
            shown = min(n, max_blocks_per_stage)
            for bi in range(shown):
                lines.append(f"      - {bi:02d}: {_short_block(st.blocks[bi])}")

            if shown < n:
                lines.append(f"      ... ({n - shown} more blocks)")

        return "\n".join(lines)
