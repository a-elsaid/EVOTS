#!/usr/bin/env python3
"""Scan run logs for architectures affected by the orphaned-stage bug (issue #3).

Before the fix, a stage i>=1 with retokenize == "none" rebuilt its tokens from the
raw input and discarded the previous stage's output. Every stage before it was
therefore trained but unable to influence the forecast — orphaned.

Two checks:
  * best genome  — reads <run>/*__best.genome.json, the reported architecture
  * whole search — reads <run>/runtime.log, counting how many evaluated genomes
                   were mis-wired (how contaminated the search itself was)

Usage:
    python tools/check_orphaned_stages.py logs/
    python tools/check_orphaned_stages.py logs/etth1_h192_quantum
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def orphaning_stages(stages: list[dict]) -> list[int]:
    """Indices of stages that refuse to listen backwards (i>=1 and retokenize=none)."""
    return [i for i, st in enumerate(stages)
            if i > 0 and st.get("retokenize") == "none"]


def check_best_genome(path: Path) -> tuple[bool, str]:
    stages = json.loads(path.read_text()).get("stages", [])
    if not stages:
        return False, "no stages recorded"

    breaks = orphaning_stages(stages)
    if not breaks:
        return False, f"{len(stages)} stage(s) — clean"

    # everything before the first non-listening stage cannot reach the head
    dead = list(range(0, breaks[0]))
    dead_blocks = sum(len(stages[i].get("blocks", [])) for i in dead)
    live_blocks = sum(len(st.get("blocks", [])) for st in stages) - dead_blocks
    dead_types = sorted({b.get("type") for i in dead for b in stages[i].get("blocks", [])})

    return True, (
        f"{len(stages)} stage(s) — stage{breaks[0]} does not listen backwards, "
        f"so stage(s) {dead} are DEAD "
        f"({dead_blocks} blocks orphaned [{', '.join(dead_types)}], {live_blocks} blocks live)"
    )


STAGE_RE = re.compile(r"\[(\d+)\] stage\d+: tokenizer=(\w+)\s+retokenize=(\w+)")
HEADER_RE = re.compile(r"Stages\((\d+)\)")


def check_search(path: Path) -> str:
    """Count how many genomes printed during the run were mis-wired."""
    total = affected = multi = 0
    current: list[tuple[int, str]] = []

    def flush():
        nonlocal total, affected, multi
        if not current:
            return
        total += 1
        if len(current) > 1:
            multi += 1
        if any(i > 0 and r == "none" for i, r in current):
            affected += 1

    for line in path.read_text(errors="ignore").splitlines():
        if HEADER_RE.search(line):
            flush()
            current = []
            continue
        m = STAGE_RE.search(line)
        if m:
            current.append((int(m.group(1)), m.group(3)))
    flush()

    if total == 0:
        return "      search: no genome printouts found in runtime.log"
    pct = 100.0 * affected / total
    return (f"      search: {affected}/{total} printed genomes mis-wired ({pct:.0f}%); "
            f"{multi} were multi-stage")


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "logs")
    if not root.exists():
        print(f"no such path: {root}", file=sys.stderr)
        return 2

    runs = sorted(p.parent for p in root.glob("*/*__best.genome.json"))
    if not runs:
        runs = [root] if list(root.glob("*__best.genome.json")) else []
    if not runs:
        print(f"no runs with a *__best.genome.json under {root}", file=sys.stderr)
        return 2

    any_bad = False
    for run in runs:
        best = next(iter(run.glob("*__best.genome.json")), None)
        if best is None:
            continue
        bad, msg = check_best_genome(best)
        any_bad |= bad
        print(f"{'AFFECTED' if bad else 'ok      '}  {run.name}: {msg}")

        log = run / "runtime.log"
        if log.exists():
            print(check_search(log))

    print()
    print("AFFECTED runs must be rerun: the reported architecture is not what trained."
          if any_bad else "All clear — no reported architecture was affected.")
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
