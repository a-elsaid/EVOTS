#!/usr/bin/env python3
"""
Build the comparison table: EvoTS vs classical baselines vs EXAQC's published
Table 1, per dataset and split mode.

Reads every results.json under a directory and reports mean +/- std, best and
worst of test accuracy and macro accuracy, with the number of runs behind each
cell. Failed and missing runs are listed explicitly -- a cell that quietly
averaged 7 of 10 seeds would read as a result when it is really a partial one.

    python tools/aggregate_classification.py --results-dir logs/classification
    python tools/aggregate_classification.py --out-dir reports/

Writes comparison.csv, comparison.md and comparison.tex.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from run_classification_suite import is_archived_run_dir  # noqa: E402

# EXAQC, arXiv 2602.03840 Table 1: per-run mean-class accuracy, in percent.
# Their protocol: 500 genomes, stratified 80/20, the 20% used as both validation
# and test, MinMax*pi scaling fitted on the full dataset. Directly comparable
# only to our "exaqc" split mode.
EXAQC_TABLE1 = {
    "iris":          [86.7, 70.0, 83.3, 70.0, 90.0],
    "seeds":         [92.9, 92.9, 95.2, 90.5, 92.9],
    "wine":          [75.0, 75.0, 77.8, 88.9, 86.1, 91.2],
    "breast_cancer": [89.2, 90.8, 88.6, 91.0],
}
EXAQC_LABEL = "EXAQC (published, 80/20, val = test)"

ALL_DATASETS = ["iris", "wine", "seeds", "breast_cancer"]
ALL_SPLIT_MODES = ["clean", "exaqc"]
DEFAULT_SEEDS = list(range(10))

EVOTS_SYSTEM = "EvoTS"
# EvoTS runs in more than one condition (classical, quantum, ...). Each becomes
# its own row, and a condition this file has never heard of appears by itself:
# the label comes from the data, not from a list here.
DEFAULT_CONDITION = "classical"
# Split modes where a baseline gets no hyperparameter selection, because
# validation is the test split there (see tools/classical_baselines.py).
UNSELECTED_MODES = {"exaqc"}


def load_results(results_dir: Path):
    """Every non-archived results.json under results_dir, plus what was skipped."""
    records, archived, unreadable = [], [], []
    for path in sorted(results_dir.rglob("results.json")):
        # A retried run's previous attempt keeps its own results.json; counting
        # it would double-count the cell it belongs to.
        if any(is_archived_run_dir(parent) for parent in path.parents):
            archived.append(path)
            continue
        try:
            records.append((path, json.loads(path.read_text())))
        except Exception as e:
            unreadable.append((path, str(e)))
    return records, archived, unreadable


def system_of(record: dict, undeclared=None) -> str:
    """
    Which system produced this run.

    NAS and baseline results share a directory, so they are told apart by
    model_family, which the baselines set explicitly. EvoTS rows are then split
    by condition, so a classical and a quantum run of the same cell never
    average together.

    A record with no condition is counted as classical and named in `undeclared`
    so the caller can say so: silently folding an unlabelled run into the
    classical row is how a mislabelled quantum result would enter the table.
    """
    if record.get("model_family") == "classical_baseline":
        return record.get("model") or "baseline-unknown"

    condition = record.get("condition")
    if not condition:
        if undeclared is not None:
            undeclared.append(record.get("run_name") or "<unnamed run>")
        condition = DEFAULT_CONDITION
    return f"{EVOTS_SYSTEM} ({condition})"


def is_evots(system: str) -> bool:
    return system.startswith(f"{EVOTS_SYSTEM} (")


def summarise(values):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "best": max(values),
        "worst": min(values),
    }


def collect(records, datasets, split_modes, seeds, undeclared=None):
    """
    cells[(dataset, split_mode, system)] -> {accuracy/macro summaries, failures}

    Every run is accounted for: ok runs feed the statistics, failed runs are
    counted and named, and seeds with no run at all are reported missing.
    """
    ok = defaultdict(lambda: {"accuracy": [], "macro": [], "seeds": set()})
    failed = defaultdict(list)

    for path, rec in records:
        dataset = Path(str(rec.get("dataset_path") or "")).stem
        mode = rec.get("split_mode")
        if dataset not in datasets or mode not in split_modes:
            continue
        key = (dataset, mode, system_of(rec, undeclared))

        if rec.get("status") != "ok":
            failed[key].append((rec.get("run_name") or path.parent.name,
                                rec.get("error") or "no error recorded",
                                str(path.parent)))
            continue

        acc, macro = rec.get("test_accuracy"), rec.get("test_macro_accuracy")
        if acc is not None:
            ok[key]["accuracy"].append(acc * 100.0)
        if macro is not None:
            ok[key]["macro"].append(macro * 100.0)
        seed = rec.get("data_split_seed")
        if seed is not None:
            ok[key]["seeds"].add(int(seed))

    cells = {}
    for key in set(ok) | set(failed):
        got = ok.get(key, {"accuracy": [], "macro": [], "seeds": set()})
        cells[key] = {
            "accuracy": summarise(got["accuracy"]),
            "macro": summarise(got["macro"]),
            "failed": failed.get(key, []),
            "missing_seeds": sorted(set(seeds) - got["seeds"]) if seeds else [],
        }
    return cells


def exaqc_cell(dataset: str):
    values = EXAQC_TABLE1.get(dataset)
    if not values:
        return None
    return {"accuracy": summarise(values), "macro": summarise(values),
            "failed": [], "missing_seeds": []}


def fmt(summary, note=""):
    if not summary:
        return "--"
    return (f"{summary['mean']:.1f} ± {summary['std']:.1f} "
            f"(best {summary['best']:.1f}, worst {summary['worst']:.1f}, "
            f"n={summary['n']}){note}")


def systems_in(cells):
    """Every system present, EvoTS conditions first (alphabetically), then the rest."""
    found = {s for (_, _, s) in cells}
    return sorted(s for s in found if is_evots(s)) + sorted(s for s in found if not is_evots(s))


def build_rows(cells, datasets, split_modes):
    """One row per (dataset, split mode, system), EvoTS first, EXAQC last."""
    rows = []
    for dataset in datasets:
        for mode in split_modes:
            ordered = systems_in(cells)
            for system in ordered:
                cell = cells.get((dataset, mode, system))
                if cell is None:
                    continue
                rows.append({
                    "dataset": dataset,
                    "split_mode": mode,
                    "system": system,
                    "comparable_to_exaqc": mode == "exaqc",
                    "cell": cell,
                    "note": (" [no tuning]" if (not is_evots(system)
                                                and mode in UNSELECTED_MODES) else ""),
                })
        if dataset in EXAQC_TABLE1:
            rows.append({
                "dataset": dataset, "split_mode": "exaqc (published)",
                "system": EXAQC_LABEL, "comparable_to_exaqc": True,
                "cell": exaqc_cell(dataset), "note": "",
            })
    return rows


def write_csv(rows, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split_mode", "system", "comparable_to_exaqc",
                    "n_runs", "accuracy_mean", "accuracy_std", "accuracy_best",
                    "accuracy_worst", "macro_mean", "macro_std", "macro_best",
                    "macro_worst", "n_failed", "missing_seeds", "notes"])
        for r in rows:
            a, m, c = r["cell"]["accuracy"], r["cell"]["macro"], r["cell"]
            w.writerow([
                r["dataset"], r["split_mode"], r["system"],
                r["comparable_to_exaqc"],
                (a or {}).get("n", 0),
                f"{a['mean']:.4f}" if a else "", f"{a['std']:.4f}" if a else "",
                f"{a['best']:.4f}" if a else "", f"{a['worst']:.4f}" if a else "",
                f"{m['mean']:.4f}" if m else "", f"{m['std']:.4f}" if m else "",
                f"{m['best']:.4f}" if m else "", f"{m['worst']:.4f}" if m else "",
                len(c["failed"]),
                " ".join(str(s) for s in c["missing_seeds"]),
                r["note"].strip(),
            ])


def write_markdown(rows, path: Path, problems):
    lines = [
        "# Classification comparison: EvoTS vs classical baselines vs EXAQC",
        "",
        "Accuracy and mean-class (macro) accuracy, in percent, as "
        "`mean ± std (best, worst, n)`.",
        "",
        "**Reading this table.** Only `exaqc` rows are comparable with the "
        "published EXAQC column: that mode reproduces their protocol "
        "(stratified 80/20, the same 20% used as both validation and test, "
        "MinMax×π scaling fitted on the full dataset). Those numbers are "
        "optimistically biased by construction, for every system in the row. "
        "`clean` rows use a stricter protocol (70/15/15, scaler fitted on train "
        "only, test evaluated once) and are **not** comparable to EXAQC's "
        "published figures.",
        "",
        "EvoTS rows are split by condition (`classical`, `quantum`, ...); a run "
        "with no condition recorded is counted as `classical` and listed under "
        "*Runs not included* so it can be checked.",
        "",
        "Baseline rows marked `[no tuning]` got no hyperparameter selection, "
        "because in that mode validation *is* test and selecting there would be "
        "selecting on test. EvoTS does select on that holdout in `exaqc` mode, "
        "so those baselines are conservative relative to it.",
        "",
    ]
    for dataset in dict.fromkeys(r["dataset"] for r in rows):
        lines += [f"## {dataset}", "",
                  "| Split mode | System | Accuracy % | Macro accuracy % | Runs | Failed | Missing seeds |",
                  "|---|---|---|---|---|---|---|"]
        for r in (x for x in rows if x["dataset"] == dataset):
            c = r["cell"]
            n = (c["accuracy"] or {}).get("n", 0)
            lines.append(
                f"| {r['split_mode']} | {r['system']}{r['note']} | "
                f"{fmt(c['accuracy'])} | {fmt(c['macro'])} | {n} | "
                f"{len(c['failed'])} | "
                f"{', '.join(str(s) for s in c['missing_seeds']) or '--'} |")
        lines.append("")

    if problems:
        lines += ["## Runs not included", ""]
        lines += problems
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_latex(rows, path: Path):
    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")

    lines = [r"% Generated by tools/aggregate_classification.py",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Dataset & System (split mode) & Accuracy \% & Macro \% & $n$ \\",
             r"\midrule"]
    last = None
    for r in rows:
        if last is not None and r["dataset"] != last:
            lines.append(r"\midrule")
        last = r["dataset"]
        a, m = r["cell"]["accuracy"], r["cell"]["macro"]
        n = (a or {}).get("n", 0)
        lines.append(
            f"{esc(r['dataset'])} & {esc(r['system'] + r['note'])} "
            f"({esc(r['split_mode'])}) & "
            + (f"{a['mean']:.1f} $\\pm$ {a['std']:.1f}" if a else "--") + " & "
            + (f"{m['mean']:.1f} $\\pm$ {m['std']:.1f}" if m else "--")
            + f" & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Aggregate classification results into a comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", default="logs/classification")
    p.add_argument("--out-dir", default=None,
                   help="where to write the tables (default: --results-dir)")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--split-modes", nargs="+", default=ALL_SPLIT_MODES,
                   choices=ALL_SPLIT_MODES)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="seeds expected per cell, for reporting missing runs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    # Check before creating anything: with out_dir defaulting to results_dir,
    # creating it first would conjure the very directory whose absence is the
    # error, and a typo'd path would report an empty table instead of failing.
    if not results_dir.exists():
        print(f"[Aggregate] No such directory: {results_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records, archived, unreadable = load_results(results_dir)
    undeclared = []
    cells = collect(records, set(args.datasets), set(args.split_modes), args.seeds,
                    undeclared=undeclared)
    rows = build_rows(cells, args.datasets, args.split_modes)

    problems = []
    for (dataset, mode, system), cell in sorted(cells.items()):
        for name, error, where in cell["failed"]:
            problems.append(f"- **failed** `{name}` ({dataset}/{mode}/{system}): "
                            f"{error} — `{where}`")
        if cell["missing_seeds"]:
            problems.append(f"- **missing** {dataset}/{mode}/{system}: no run for "
                            f"seeds {', '.join(str(s) for s in cell['missing_seeds'])}")
    for path, error in unreadable:
        problems.append(f"- **unreadable** `{path}`: {error}")
    if undeclared:
        problems.append(
            f"- **no condition recorded** on {len(undeclared)} EvoTS run(s), counted "
            f"as `{DEFAULT_CONDITION}`: {', '.join(sorted(set(undeclared))[:10])}"
            + (" ..." if len(set(undeclared)) > 10 else "")
            + ". Runs launched through the suite record one; these were not.")

    write_csv(rows, out_dir / "comparison.csv")
    write_markdown(rows, out_dir / "comparison.md", problems)
    write_latex(rows, out_dir / "comparison.tex")

    print(f"[Aggregate] {len(records)} results read "
          f"({len(archived)} archived attempts skipped, {len(unreadable)} unreadable)")
    print(f"[Aggregate] Wrote {out_dir/'comparison.csv'}, "
          f"{out_dir/'comparison.md'}, {out_dir/'comparison.tex'}")
    if problems:
        print(f"[Aggregate] {len(problems)} runs not included:")
        for p in problems:
            print("   " + p.replace("**", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
