#!/usr/bin/env python3
"""Time QuantumMixBlock across encoding / readout / qubit count.

Exists to pick quantum_amplitude_qubits_range and quantum_angle_qubits_range in
SearchSpaceConfig empirically, rather than guessing. State-vector simulation costs
2^n complex amplitudes PER TOKEN, vmapped over B*N tokens, so the ceiling is real
and hard: an over-wide range makes search workers OOM, and a worker OOM surfaces as
inf fitness — indistinguishable from a genuinely bad architecture.

Two things this measures that a naive timing loop would miss:

  * Compile and steady-state times are reported separately. Every distinct circuit
    configuration compiles its own TF graph through torch_interface(jit=True). Today
    there is one configuration per process so that cost is paid once and forgotten.
    Once encoding/readout/n_qubits are genes, each distinct genome pays it again, and
    across a few hundred evaluations compile time can dominate wall clock.

  * n_qubits is driven independently of d_model via the input projection. Amplitude
    encoding without a projection pins n to log2(d_model), so every amplitude row off
    that diagonal would otherwise be unmeasurable.

out_proj parameter count is printed alongside, because it grows as 2^n * d_model for
the state and prob readouts: at d_model=256, n=14 that single layer is 4.2M
parameters, larger than the rest of the block combined. expval_z stays at n*d_model.

Usage:
    python tools/bench_quantum.py
    python tools/bench_quantum.py --qubits 6,8,10,12 --batch 16 --tokens 32
    python tools/bench_quantum.py --device cuda --backward
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback

import torch

from nas_ts.models.model_builder_v2 import (
    QUANTUM_ENCODINGS,
    QUANTUM_READOUTS,
    QuantumMixBlock,
)


def _params(module) -> int:
    return 0 if module is None else sum(p.numel() for p in module.parameters())


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def bench_one(
    encoding: str,
    readout: str,
    n_qubits: int,
    *,
    d_model: int,
    batch: int,
    tokens: int,
    repeats: int,
    device: torch.device,
    backward: bool,
) -> dict:
    """Build one block, time the first (compiling) call and then steady state."""
    torch.manual_seed(0)
    block = QuantumMixBlock(
        d_model=d_model,
        nlayers=2,
        entangle_pattern="linear",
        gate_set="rx_ry",
        use_ffn=False,
        ff_mult=2.0,
        dropout=0.0,
        encoding=encoding,
        readout=readout,
        n_qubits=n_qubits,
        reupload=False,
    ).to(device)
    block.train() if backward else block.eval()

    x = torch.randn(batch, tokens, d_model, device=device)

    def _run():
        if backward:
            block.zero_grad(set_to_none=True)
            block(x).square().mean().backward()
        else:
            with torch.no_grad():
                block(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    _run()  # includes circuit construction + TF graph compilation
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(repeats):
        _run()
    steady_s = (time.perf_counter() - t0) / repeats

    peak_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _run()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    return {
        "compile_s": compile_s,
        "steady_s": steady_s,
        "peak_mb": peak_mb,
        "readout_width": block.readout_width,
        "out_proj_params": _params(block.out_proj),
        "in_proj_params": _params(block.in_proj),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qubits", default="6,8,9,10,12,14",
                    help="comma-separated qubit counts (default: 6,8,9,10,12,14)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens per sample; total vmap width is batch*tokens")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backward", action="store_true",
                    help="time forward+backward instead of forward only")
    args = ap.parse_args(argv)

    qubits = [int(q) for q in args.qubits.split(",") if q.strip()]
    device = torch.device(args.device)

    print(f"device={device}  d_model={args.d_model}  batch={args.batch}  "
          f"tokens={args.tokens}  vmap_width={args.batch * args.tokens}  "
          f"pass={'fwd+bwd' if args.backward else 'fwd'}  repeats={args.repeats}")
    print()
    header = (f"{'enc':<10}{'readout':<10}{'n':>3}{'width':>8}{'out_proj':>10}"
              f"{'compile_s':>11}{'steady_s':>10}{'peak_MB':>10}")
    print(header)
    print("-" * len(header))

    for n in qubits:
        for encoding in QUANTUM_ENCODINGS:
            for readout in QUANTUM_READOUTS:
                try:
                    r = bench_one(
                        encoding, readout, n,
                        d_model=args.d_model, batch=args.batch, tokens=args.tokens,
                        repeats=args.repeats, device=device, backward=args.backward,
                    )
                except Exception as e:  # OOM and tc failures are the point of the sweep
                    print(f"{encoding:<10}{readout:<10}{n:>3}{'':>8}{'':>10}"
                          f"{'':>11}{'FAILED':>10}   {type(e).__name__}: {e}")
                    traceback.print_exc(limit=1, file=sys.stderr)
                    continue
                peak = "-" if r["peak_mb"] is None else f"{r['peak_mb']:.0f}"
                print(f"{encoding:<10}{readout:<10}{n:>3}{r['readout_width']:>8}"
                      f"{_fmt_count(r['out_proj_params']):>10}"
                      f"{r['compile_s']:>11.2f}{r['steady_s']:>10.4f}{peak:>10}")
        print()

    print("Notes:")
    print("  width      = readout width per token (2^n for state/prob, n for expval_z)")
    print("  out_proj   = params in the readout->d_model projection (0 when widths match)")
    print("  compile_s  = first call: circuit construction + TF graph compile, paid once")
    print("               per distinct configuration, i.e. once per distinct genome")
    print("  steady_s   = mean of subsequent calls, JIT warm")
    print()
    print(f"  n = log2(d_model) = {int(math.log2(args.d_model))} is the amplitude path that")
    print("  needs no input projection; every other cell pays for one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
