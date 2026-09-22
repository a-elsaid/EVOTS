#!/usr/bin/env python3
"""
Run the full classification experiment suite: every dataset x split mode x seed.

Each run is a separate subprocess. That is the point: a worker dying natively
(BrokenProcessPool, an OOM kill, a CUDA fault) takes down one run, not the
sweep. The suite records what happened and carries on.

Resumable by design -- a run whose results.json already says status "ok" is
skipped -- so this can be re-invoked after a walltime kill and it will pick up
where it stopped.

    python experiments/run_classification_suite.py
    python experiments/run_classification_suite.py --datasets iris wine --seeds 0 1 2
    python experiments/run_classification_suite.py --split-modes exaqc
    python experiments/run_classification_suite.py --dry-run

Exit code is non-zero if any run failed, so a scheduler notices.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nas_ts.utils.run_results import (  # noqa: E402
    RESULTS_FILENAME, STATUS_FAILED, STATUS_OK, write_results_json,
)

ALL_DATASETS = ["iris", "wine", "seeds", "breast_cancer"]
ALL_SPLIT_MODES = ["clean", "exaqc"]
DEFAULT_SEEDS = list(range(10))

CONFIG_DIR = REPO_ROOT / "configs" / "classification"
DATA_DIR = REPO_ROOT / "data" / "tabular"
FETCH_SCRIPT = REPO_ROOT / "fetch_datasets.py"

# Overrides the suite owns. A user --set on one of these would break the
# identity of the sweep (two runs writing to one directory, or a seed that does
# not match its name), so they are applied last and a collision is called out.
OWNED_KEYS = {
    "run.name", "run.logs_dir", "run.checkpoints_dir",
    "evo.random_seed", "data.tabular.random_seed", "data.tabular.split_mode",
}

# A re-executed run's previous directory is moved aside under this suffix, so a
# fresh attempt never starts on top of a half-finished one and the old attempt
# survives as evidence.
#
# AGGREGATION: these directories still contain a results.json, so any tool that
# globs for results.json MUST skip paths where is_archived_run_dir() is true.
# Otherwise a failed first attempt would be counted alongside the run that
# replaced it, double-counting the cell.
ATTEMPT_SUFFIX = ".attempt-"
_ARCHIVED_RE = re.compile(r"\.attempt-\d{8}T\d{6}Z(-\d+)?$")


def is_archived_run_dir(path) -> bool:
    """True for a superseded attempt directory (see ATTEMPT_SUFFIX)."""
    return bool(_ARCHIVED_RE.search(Path(path).name))


def archive_previous_attempt(rdir: Path):
    """
    Move an existing run directory aside, returning where it went.

    A retry that reuses the directory would mix two attempts' logs, plots and
    checkpoints, and leave the previous results.json in place until the new run
    happens to overwrite it.
    """
    if not rdir.exists():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = rdir.with_name(rdir.name + ATTEMPT_SUFFIX + stamp)
    n = 1
    while dest.exists():  # two attempts within the same second
        dest = rdir.with_name(f"{rdir.name}{ATTEMPT_SUFFIX}{stamp}-{n}")
        n += 1
    rdir.rename(dest)
    return dest


def run_name(dataset: str, split_mode: str, seed: int) -> str:
    return f"{dataset}_{split_mode}_seed{seed}"


def run_dir(out_dir: Path, name: str) -> Path:
    return out_dir / name


def results_file(out_dir: Path, name: str) -> Path:
    return run_dir(out_dir, name) / RESULTS_FILENAME


def already_ok(path: Path) -> bool:
    """True only for a readable results.json that says status ok."""
    try:
        return json.loads(path.read_text()).get("status") == STATUS_OK
    except Exception:
        # Missing, truncated, or mid-write: treat as not done and rerun.
        return False


def ensure_datasets(datasets) -> None:
    """
    Make sure every needed CSV is present, fetching once if not.

    Three of the four come from scikit-learn and work offline; seeds is pulled
    from UCI and needs network, which a compute node often does not have. So a
    partial fetch is the normal failure, and the message has to name the files
    that are still missing and where they go.
    """
    missing = [d for d in datasets if not (DATA_DIR / f"{d}.csv").exists()]
    if not missing:
        return

    print(f"[Suite] Missing dataset CSVs: {', '.join(missing)}. Running {FETCH_SCRIPT.name} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, str(FETCH_SCRIPT)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0:
            print(f"[Suite] {FETCH_SCRIPT.name} exited {proc.returncode}:", file=sys.stderr)
            print((proc.stderr or "").rstrip()[-2000:], file=sys.stderr)
    except Exception as e:
        print(f"[Suite] Could not run {FETCH_SCRIPT.name}: {e}", file=sys.stderr)

    still_missing = [d for d in datasets if not (DATA_DIR / f"{d}.csv").exists()]
    if still_missing:
        print(
            "\n[Suite] STOPPING: these dataset files are still missing:\n"
            + "".join(f"    {DATA_DIR / (d + '.csv')}\n" for d in still_missing)
            + "\nThe fetch needs network access for 'seeds' (UCI); iris, wine and\n"
              "breast_cancer come from scikit-learn and work offline.\n"
              "On a machine with network, run:\n"
              f"    python {FETCH_SCRIPT.name}\n"
              "then copy the file(s) above to the same path on this machine, e.g.:\n"
              f"    scp {' '.join(d + '.csv' for d in still_missing)} "
              f"<this-host>:{DATA_DIR}/\n",
            file=sys.stderr,
        )
        sys.exit(2)


def build_command(dataset: str, split_mode: str, seed: int, out_dir: Path,
                  user_sets) -> list:
    config = CONFIG_DIR / f"{dataset}.yml"
    name = run_name(dataset, split_mode, seed)

    cmd = [sys.executable, str(REPO_ROOT / "experiments" / "run_exp.py"),
           "--config", str(config)]
    # User overrides first, ours last: load_cfg applies in order and last wins,
    # so the sweep's identity cannot be overridden by accident.
    for s in user_sets or []:
        cmd += ["--set", s]
    for s in (
        f"run.name={name}",
        f"run.logs_dir={out_dir}",
        # Per run: the config's shared checkpoints/ would have every run writing
        # ckpt_*.pt over every other run's.
        f"run.checkpoints_dir={run_dir(out_dir, name) / 'checkpoints'}",
        f"evo.random_seed={seed}",
        f"data.tabular.random_seed={seed}",
        f"data.tabular.split_mode={split_mode}",
    ):
        cmd += ["--set", s]
    return cmd


def record_crash(out_dir: Path, dataset: str, split_mode: str, seed: int,
                 returncode, log_path: Path, elapsed: float) -> None:
    """
    Write a failed results.json for a run that died without writing one.

    Without this a natively-killed run is indistinguishable from one that never
    started, and the aggregation step would silently omit it.
    """
    name = run_name(dataset, split_mode, seed)
    write_results_json(results_file(out_dir, name), {
        "status": STATUS_FAILED,
        "run_name": name,
        "config_path": str(CONFIG_DIR / f"{dataset}.yml"),
        "task_type": "classification",
        "dataset_path": str(DATA_DIR / f"{dataset}.csv"),
        "split_mode": split_mode,
        "evo_random_seed": seed,
        "data_split_seed": seed,
        "wall_clock_seconds": round(elapsed, 3),
        "error": (f"run_exp exited {returncode} without writing {RESULTS_FILENAME}; "
                  f"see {log_path}"),
        "recorded_by": "run_classification_suite",
    })


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run every classification experiment (dataset x split mode x seed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--split-modes", nargs="+", default=ALL_SPLIT_MODES,
                   choices=ALL_SPLIT_MODES)
    p.add_argument("--out-dir", default="logs/classification",
                   help="where each run's directory and results.json are written")
    p.add_argument("--set", action="append", dest="sets", default=[],
                   metavar="key.path=value",
                   help="passed through to run_exp; repeatable")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-run timeout in seconds (default: none)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if results.json already says ok")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would run, then exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    clobbered = sorted(k for k in OWNED_KEYS
                       if any(s.split("=", 1)[0].strip() == k for s in args.sets))
    if clobbered:
        print(f"[Suite] NOTE: --set on {', '.join(clobbered)} is ignored; "
              f"the suite sets these per run.")

    plan = [(d, m, s)
            for d in args.datasets
            for m in args.split_modes
            for s in args.seeds]

    print(f"[Suite] {len(plan)} runs: {len(args.datasets)} datasets x "
          f"{len(args.split_modes)} split modes x {len(args.seeds)} seeds")
    print(f"[Suite] Output: {out_dir.resolve()}")

    if args.dry_run:
        for d, m, s in plan:
            name = run_name(d, m, s)
            state = "skip (ok)" if (not args.force and already_ok(results_file(out_dir, name))) else "run"
            print(f"    {name:44s} {state}")
        return 0

    ensure_datasets(args.datasets)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = [], [], []
    suite_started = time.perf_counter()

    for i, (dataset, split_mode, seed) in enumerate(plan, start=1):
        name = run_name(dataset, split_mode, seed)
        rpath = results_file(out_dir, name)

        if not args.force and already_ok(rpath):
            skipped.append(name)
            print(f"[{i}/{len(plan)}] {name}: skip, already ok")
            continue

        rdir = run_dir(out_dir, name)
        archived = archive_previous_attempt(rdir)
        if archived is not None:
            print(f"    previous attempt moved to {archived.name}")
        rdir.mkdir(parents=True, exist_ok=True)
        log_path = rdir / "suite_run.log"

        cmd = build_command(dataset, split_mode, seed, out_dir, args.sets)
        print(f"[{i}/{len(plan)}] {name}: running -> {log_path}")

        started = time.perf_counter()
        returncode = None
        try:
            with open(log_path, "w") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log,
                                      stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
            print(f"    timed out after {args.timeout}s")
        except Exception as e:
            returncode = f"launch-error: {e}"
            print(f"    could not launch: {e}")

        elapsed = time.perf_counter() - started

        if already_ok(rpath):
            ok.append(name)
            print(f"    ok in {elapsed:.1f}s")
        else:
            if not rpath.exists():
                record_crash(out_dir, dataset, split_mode, seed, returncode, log_path, elapsed)
            failed.append((name, str(log_path)))
            print(f"    FAILED (exit={returncode}) after {elapsed:.1f}s")

    total = time.perf_counter() - suite_started
    print("\n" + "=" * 70)
    print(f"[Suite] {len(ok)} ok | {len(failed)} failed | {len(skipped)} skipped "
          f"| {len(plan)} planned | {total:.1f}s")
    if failed:
        print("[Suite] Failed runs:")
        for name, log in failed:
            print(f"    {name:44s} {log}")
        print("[Suite] Re-run the suite to retry them; runs already ok are skipped.")
    print("=" * 70)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
