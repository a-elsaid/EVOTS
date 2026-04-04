from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import TaskConfig
from ..core.genome_v2 import Genome, BlockSpec, StageSpec, PatchingSpec


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:])


def _compatible_num_heads(dim: int, requested_heads: int) -> int:
    requested_heads = max(1, int(requested_heads))
    dim = int(dim)
    for h in range(min(requested_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


class WindowNorm(nn.Module):
    """Per-window, per-variable normalization matching iTransformer's non-stationary norm."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = x.mean(dim=1, keepdim=True).detach()
        x0 = x - means
        stdev = torch.sqrt(torch.var(x0, dim=1, keepdim=True, unbiased=False) + self.eps)
        return x0 / stdev, means, stdev

    def denorm(self, y: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        d_model = int(d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TimeTokenHead(nn.Module):
    """Projects [B, L, d_in] -> [B, L, D]."""

    def __init__(self, d_in: int, D: int, pos_encoding: str = "none", dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(int(d_in), int(D))
        self.pe = PositionalEncoding(int(D)) if pos_encoding == "absolute" else None
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.proj(x)
        if self.pe is not None:
            z = self.pe(z)
        return self.drop(z)


class PatchTokenHead(nn.Module):
    """Projects [B, L, d_in] -> [B, Npatch, D] via temporal patching."""

    def __init__(self, d_in: int, D: int, patching: PatchingSpec, dropout: float = 0.0):
        super().__init__()
        self.p = int(patching.patch_size)
        self.s = int(patching.stride) if int(patching.stride) > 0 else int(patching.patch_size)
        self.D = int(D)
        self.proj: Optional[nn.Linear] = None
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, Din = x.shape
        p, s = self.p, self.s

        if L < p:
            x = F.pad(x, (0, 0, 0, p - L))
            L = x.size(1)

        rem = (L - p) % s
        if rem != 0:
            x = F.pad(x, (0, 0, 0, s - rem))

        patches = x.unfold(dimension=1, size=p, step=s)
        B2, N, p2, Din2 = patches.shape
        flat = patches.reshape(B2, N, p2 * Din2)

        if self.proj is None or self.proj.in_features != flat.size(-1):
            self.proj = nn.Linear(flat.size(-1), self.D).to(x.device)

        return self.drop(self.proj(flat))


class VarTokenHead_iTransformer(nn.Module):
    """
    Matches iTransformer DataEmbedding_inverted:
    x: [B, L, N] -> permute -> [B, N, L] -> Linear(L -> D)
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.0, include_mark_tokens: bool = False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.include_mark_tokens = bool(include_mark_tokens)
        self.value_embedding = nn.Linear(self.seq_len, self.d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, N = x.shape
        if L != self.seq_len:
            raise ValueError(f"VarTokenHead expected L={self.seq_len}, got {L}")

        v = x.permute(0, 2, 1).contiguous()

        if self.include_mark_tokens and x_mark is not None:
            m = x_mark.permute(0, 2, 1).contiguous()
            v = torch.cat([v, m], dim=1)

        return self.drop(self.value_embedding(v))


class CrossTokenHead(nn.Module):
    """Crossformer-like tokenization: [B, L, d_in] -> [B, G*Nseg, D]."""

    def __init__(self, d_in: int, D: int, cross_head_spec, dropout: float = 0.0):
        super().__init__()
        self.D = int(D)
        self.groups = int(max(1, cross_head_spec.groups))
        self.patch_size = int(cross_head_spec.patch_size)
        self.stride = int(cross_head_spec.stride if cross_head_spec.stride > 0 else cross_head_spec.patch_size)
        self.encoder_type = getattr(cross_head_spec, "encoder_type", "linear")
        self.pool = getattr(cross_head_spec, "pool", "avg")

        self.proj_linear: Optional[nn.Linear] = None
        k = int(getattr(cross_head_spec, "conv_kernel", 3))
        self.conv = nn.Conv1d(1, 8, kernel_size=k, padding=k // 2)
        self.conv_proj = nn.Linear(8, self.D)
        self.ln = nn.LayerNorm(self.D)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, Din = x.shape
        p, s = self.patch_size, self.stride

        if L < p:
            x = F.pad(x, (0, 0, 0, p - L))
            L = x.size(1)

        rem = (L - p) % s
        if rem != 0:
            x = F.pad(x, (0, 0, 0, s - rem))
            L = x.size(1)

        G = self.groups
        group_width = math.ceil(Din / G)
        target_d = group_width * G
        if target_d != Din:
            x = F.pad(x, (0, target_d - Din, 0, 0))

        xg = x.view(B, L, G, group_width)
        seg = xg.unfold(dimension=1, size=p, step=s)
        B2, Nseg, G2, W2, p2 = seg.shape

        seg = seg.permute(0, 2, 1, 4, 3).contiguous()
        tokens_raw = seg.reshape(B * G2 * Nseg, p2, W2)

        if self.encoder_type == "conv":
            collapsed = tokens_raw.mean(dim=2).unsqueeze(1)
            z = self.conv(collapsed)
            z = z.max(dim=2).values if self.pool == "max" else z.mean(dim=2)
            tok = self.conv_proj(z)
        else:
            pooled = tokens_raw.mean(dim=1)
            if self.proj_linear is None or self.proj_linear.in_features != pooled.size(-1):
                self.proj_linear = nn.Linear(pooled.size(-1), self.D).to(x.device)
            tok = self.proj_linear(pooled)

        tok = self.drop(self.ln(tok))
        return tok.view(B, G * Nseg, self.D)


class FullAttentionRepo(nn.Module):
    """Repo-like full attention over tokens (no causal mask)."""

    def __init__(self, attention_dropout: float = 0.1, scale: Optional[float] = None):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(float(attention_dropout))

    def forward(self, queries, keys, values, attn_mask=None):
        B, Lq, H, E = queries.shape
        scale = self.scale or (1.0 / math.sqrt(E))
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous(), None


class AttentionLayerRepo(nn.Module):
    """Repo-like attention layer with explicit QKV projections."""

    def __init__(self, d_model: int, n_heads: int, attention_dropout: float = 0.1):
        super().__init__()
        d_model = int(d_model)
        n_heads = int(n_heads)
        assert d_model % n_heads == 0
        d_head = d_model // n_heads

        self.n_heads = n_heads
        self.d_head = d_head
        self.inner_attention = FullAttentionRepo(attention_dropout=attention_dropout)
        self.query_projection = nn.Linear(d_model, d_head * n_heads)
        self.key_projection = nn.Linear(d_model, d_head * n_heads)
        self.value_projection = nn.Linear(d_model, d_head * n_heads)
        self.out_projection = nn.Linear(d_head * n_heads, d_model)

    def forward(self, x: torch.Tensor, attn_mask=None):
        B, L, D = x.shape
        H = self.n_heads
        q = self.query_projection(x).view(B, L, H, self.d_head)
        k = self.key_projection(x).view(B, L, H, self.d_head)
        v = self.value_projection(x).view(B, L, H, self.d_head)
        out, _ = self.inner_attention(q, k, v, attn_mask=attn_mask)
        return self.out_projection(out.view(B, L, H * self.d_head)), None


class FeedForwardConv1x1Repo(nn.Module):
    """Repo-style FFN using Conv1d(kernel=1) matching iTransformer EncoderLayer."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.conv1 = nn.Conv1d(int(d_model), int(d_ff), kernel_size=1)
        self.conv2 = nn.Conv1d(int(d_ff), int(d_model), kernel_size=1)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = F.gelu if activation.lower() == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.conv1(x.transpose(1, 2)))
        y = self.dropout(self.conv2(self.dropout(y)))
        return y.transpose(1, 2)


class ITransformerEncoderBlock(nn.Module):
    """Repo-faithful iTransformer encoder block (inv_attn block type)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        d_model = int(d_model)
        dropout = float(dropout)
        self.attention = AttentionLayerRepo(d_model=d_model, n_heads=n_heads, attention_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardConv1x1Repo(d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x, _ = self.attention(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        return self.norm2(x + self.ffn(y))


class StandardAttnBlock(nn.Module):
    """Generic pre-LN Transformer encoder block (attn block type)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: float, dropout: float):
        super().__init__()
        d_model = int(d_model)
        n_heads = _compatible_num_heads(d_model, n_heads)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        hidden = int(d_model * float(ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(_ln(x), _ln(x), x)
        x = x + self.drop(a)
        return x + self.drop(self.ff(_ln(x)))


class ConvTokenMixBlock(nn.Module):
    """Depthwise conv over token axis followed by pointwise projection (conv block type)."""

    def __init__(self, d_model: int, kernel_size: int, ff_mult: float, dropout: float):
        super().__init__()
        d_model = int(d_model)
        k = int(kernel_size)
        self.norm = nn.LayerNorm(d_model)
        self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(float(dropout))
        hidden = int(d_model * float(ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.pw(self.dwconv(h.transpose(1, 2)).transpose(1, 2))
        x = x + self.drop(y)
        return x + self.drop(self.ff(_ln(x)))


def _block_dim(spec: BlockSpec, genome: Genome) -> int:
    return int(spec.dim or genome.model_dim)

def _block_heads(spec: BlockSpec, genome: Genome) -> int:
    return int(spec.num_heads or genome.num_heads)

def _block_ff_mult(spec: BlockSpec, genome: Genome) -> float:
    return float(spec.ff_mult or genome.ff_mult)


def make_block(spec: BlockSpec, genome: Genome, *, activation: str = "gelu") -> nn.Module:
    bt = (spec.block_type or "attn").lower()
    d_model = _block_dim(spec, genome)
    dropout = float(genome.dropout)

    if bt in {"inv_attn", "itransformer", "it", "inverted_attn"}:
        n_heads = _compatible_num_heads(d_model, _block_heads(spec, genome))
        d_ff = int(d_model * _block_ff_mult(spec, genome))
        return ITransformerEncoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, activation=activation)

    if bt in {"attn", "mha", "self_attn"}:
        n_heads = _compatible_num_heads(d_model, _block_heads(spec, genome))
        return StandardAttnBlock(d_model=d_model, n_heads=n_heads, ff_mult=_block_ff_mult(spec, genome), dropout=dropout)

    if bt in {"conv", "conv1d"}:
        conv_spec = getattr(genome, "conv_block", None)
        kernel = int(getattr(conv_spec, "kernel_size", 3)) if conv_spec is not None else 3
        return ConvTokenMixBlock(d_model=d_model, kernel_size=kernel, ff_mult=_block_ff_mult(spec, genome), dropout=dropout)

    if bt in {"freq", "fft", "decomp", "cross_dim", "crossdim"}:
        return nn.Identity()

    raise ValueError(f"Unknown block_type={spec.block_type!r}")


class TransformerStack(nn.Module):
    def __init__(self, genome: Genome, blocks: Sequence[BlockSpec], *, activation: str = "gelu"):
        super().__init__()
        if not blocks:
            blocks = [BlockSpec(block_type="attn")]
        self.layers = nn.ModuleList([make_block(b, genome, activation=activation) for b in blocks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttentionRetokenizer(nn.Module):
    """Builds new stage tokens using raw time tokens + previous stage tokens as memory."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        d_model = int(d_model)
        heads = _compatible_num_heads(d_model, num_heads)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.drop = nn.Dropout(float(dropout))
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        base_next_tokens: torch.Tensor,
        raw_time_tokens: torch.Tensor,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = _ln(base_next_tokens)
        mem = raw_time_tokens if prev_tokens is None else torch.cat([prev_tokens, raw_time_tokens], dim=1)
        a, _ = self.attn(q, _ln(mem), mem)
        x = base_next_tokens + self.drop(a)
        return x + self.drop(self.ff(_ln(x)))


class ITransformerProjectorHead(nn.Module):
    """
    Repo-style projector head:
    [B, N_tokens, D] -> Linear(D -> pred_len) -> [B, pred_len, N_vars]
    """

    def __init__(self, d_model: int, pred_len: int):
        super().__init__()
        self.projector = nn.Linear(int(d_model), int(pred_len), bias=True)

    def forward(self, enc_out: torch.Tensor, *, n_vars: int) -> torch.Tensor:
        y = self.projector(enc_out).permute(0, 2, 1).contiguous()
        return y[:, :, :n_vars]


class TokenPoolForecastHead(nn.Module):
    """Pools over token axis then projects to [B, pred_len, d_out]."""

    def __init__(self, d_model: int, pred_len: int, d_out: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.pred_len = int(pred_len)
        self.d_out = int(d_out)
        self.proj = nn.Linear(int(d_model), self.pred_len * self.d_out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = z.max(dim=1).values if self.pool == "max" else z.mean(dim=1)
        return self.proj(p).view(z.size(0), self.pred_len, self.d_out)


def make_tokenizer(task: TaskConfig, genome: Genome, stage: StageSpec) -> nn.Module:
    tok = stage.tokenizer

    if tok == "time":
        return TimeTokenHead(task.d_in, genome.model_dim, pos_encoding=genome.pos_encoding, dropout=genome.dropout)

    if tok == "patch":
        return PatchTokenHead(
            task.d_in, genome.model_dim,
            PatchingSpec(enabled=True, patch_size=16, stride=16),
            dropout=genome.dropout,
        )

    if tok == "var":
        return VarTokenHead_iTransformer(
            seq_len=task.input_length,
            d_model=genome.model_dim,
            dropout=genome.dropout,
            include_mark_tokens=False,
        )

    if tok == "cross":
        ch = genome.cross_head
        ch.enabled = True
        return CrossTokenHead(task.d_in, genome.model_dim, ch, dropout=genome.dropout)

    raise ValueError(f"Unknown stage.tokenizer={tok!r}")


class StagedForecastModel(nn.Module):
    """
    Stage-based forecasting model:
      (optional per-window norm) -> stages(tokenize -> optional retokenize -> core) -> head -> (optional denorm)
    """

    def __init__(self, genome: Genome, task: TaskConfig):
        super().__init__()
        self.genome = genome
        self.task = task

        self.use_norm = bool(getattr(task, "use_norm", False))
        self.normer = WindowNorm() if self.use_norm else None

        self.raw_time = TimeTokenHead(task.d_in, genome.model_dim, pos_encoding="none", dropout=genome.dropout)

        self.stage_modules = nn.ModuleList()
        for st in genome.stages:
            tok = make_tokenizer(task, genome, st)
            core = TransformerStack(genome, st.blocks, activation="gelu")
            retok = CrossAttentionRetokenizer(genome.model_dim, genome.num_heads, genome.dropout) \
                if st.retokenize == "cross_attn" else nn.Identity()
            self.stage_modules.append(nn.ModuleDict({"tok": tok, "retok": retok, "core": core}))

        if genome.stages[0].tokenizer == "var":
            self.head = ITransformerProjectorHead(genome.model_dim, task.pred_length)
        else:
            self.head = TokenPoolForecastHead(genome.model_dim, task.pred_length, task.d_out, pool="mean")

    def forecast(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        means = stdev = None
        if self.use_norm and self.normer is not None:
            x, means, stdev = self.normer(x)

        raw_time_tokens = self.raw_time(x)
        prev_tokens: Optional[torch.Tensor] = None
        tokens: Optional[torch.Tensor] = None

        for mod in self.stage_modules:
            base = mod["tok"](x, x_mark=x_mark)
            if not isinstance(mod["retok"], nn.Identity):
                base = mod["retok"](base, raw_time_tokens, prev_tokens)
            tokens = mod["core"](base)
            prev_tokens = tokens

        assert tokens is not None

        if isinstance(self.head, ITransformerProjectorHead):
            y = self.head(tokens, n_vars=int(self.task.d_in))
        else:
            y = self.head(tokens)

        if self.use_norm and self.normer is not None and means is not None and stdev is not None:
            if y.ndim == 3 and y.size(-1) == x.size(-1):
                y = self.normer.denorm(y, means, stdev)

        return y

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_mark: Optional[torch.Tensor] = None,
        y_mark: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forecast(x, x_mark=x_mark)


def resolve_task(task: TaskConfig, *, d_in: Optional[int] = None, d_out: Optional[int] = None) -> TaskConfig:
    return TaskConfig(
        task_type=task.task_type,
        input_length=task.input_length,
        pred_length=task.pred_length,
        d_in=task.d_in if d_in is None else int(d_in),
        d_out=task.d_out if d_out is None else int(d_out),
        metrics=task.metrics,
    )


def build_model(genome: Genome, task: TaskConfig, *, d_in: Optional[int] = None, d_out: Optional[int] = None) -> nn.Module:
    t = resolve_task(task, d_in=d_in, d_out=d_out)
    if t.task_type != "forecasting":
        raise NotImplementedError("Only forecasting supported.")
    return StagedForecastModel(genome, t)


def build_model_from_meta(genome: Genome, task: TaskConfig, meta: Mapping[str, Any]) -> nn.Module:
    d_in = int(meta.get("d_in", task.d_in))
    d_out = int(meta.get("d_out", task.d_out))
    return build_model(genome, task, d_in=d_in, d_out=d_out)