"""
The table must show whether a quantum condition actually used quantum.

Nothing requires a genome to contain a quantum block, so a quantum-condition row
can be entirely classical architectures found under a quantum budget. Reading
that row as a result about quantum circuits would be wrong, and the config
directory that produced it is not visible in the table -- so the count has to be.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import aggregate_classification as agg  # noqa: E402


def _genome(block_types):
    """A best_genome dict with the given block types across two stages."""
    half = len(block_types) // 2
    return {"family": "iT", "stages": [
        {"blocks": [{"block_type": b} for b in block_types[:half]]},
        {"blocks": [{"block_type": b} for b in block_types[half:]]},
    ]}


def write_result(root: Path, name: str, **fields):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok", "run_name": name,
        "dataset_path": "data/tabular/iris.csv", "split_mode": "clean",
        "data_split_seed": 0, "test_accuracy": 0.9, "test_macro_accuracy": 0.9,
        "condition": "quantum",
    }
    payload.update(fields)
    (d / "results.json").write_text(json.dumps(payload))


# ------------------------------------------------------------- the counter

@pytest.mark.parametrize("blocks,expected", [
    (["attn", "quantum", "conv", "quantum"], 2),
    (["attn", "conv", "attn", "conv"], 0),
    (["quantum", "quantum", "quantum", "quantum"], 4),
])
def test_counts_quantum_blocks_in_the_winning_genome(blocks, expected):
    assert agg.count_quantum_blocks({"best_genome": _genome(blocks)}) == expected


def test_returns_none_when_there_is_no_genome():
    """None means unknowable; 0 means a genome with no quantum block."""
    assert agg.count_quantum_blocks({}) is None
    assert agg.count_quantum_blocks({"best_genome": None}) is None
    assert agg.count_quantum_blocks({"best_genome": {"stages": "not a list"}}) is None
    assert agg.count_quantum_blocks({"best_genome": _genome(["attn"])}) == 0


def test_tolerates_a_malformed_genome():
    """A results.json from another tool must not crash the aggregation."""
    assert agg.count_quantum_blocks({"best_genome": {"stages": [None, {}]}}) == 0
    assert agg.count_quantum_blocks({"best_genome": {"stages": [{"blocks": None}]}}) == 0


# --------------------------------------------------------------- the cell

def test_cell_reports_mean_and_zero_fraction(tmp_path):
    write_result(tmp_path, "quantum_iris_clean_seed0", data_split_seed=0,
                 best_genome=_genome(["attn", "quantum", "conv", "quantum"]))   # 2
    write_result(tmp_path, "quantum_iris_clean_seed1", data_split_seed=1,
                 best_genome=_genome(["attn", "conv", "attn", "conv"]))         # 0
    write_result(tmp_path, "quantum_iris_clean_seed2", data_split_seed=2,
                 best_genome=_genome(["quantum", "conv", "attn", "conv"]))      # 1

    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean"}, [0, 1, 2])
    q = cells[("iris", "clean", "EvoTS (quantum)")]["quantum"]

    assert q["n"] == 3
    assert q["mean"] == pytest.approx(1.0)
    assert q["zero_fraction"] == pytest.approx(1 / 3)


def test_a_wholly_classical_quantum_row_is_visible(tmp_path):
    """The finding this column exists for."""
    for seed in range(4):
        write_result(tmp_path, f"quantum_iris_clean_seed{seed}", data_split_seed=seed,
                     best_genome=_genome(["attn", "conv", "attn", "conv"]))
    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean"}, list(range(4)))
    cell = cells[("iris", "clean", "EvoTS (quantum)")]

    assert cell["quantum"]["mean"] == 0.0
    assert cell["quantum"]["zero_fraction"] == 1.0
    assert "100% none" in agg.fmt_quantum(cell)


def test_failed_runs_do_not_contribute_a_count(tmp_path):
    write_result(tmp_path, "quantum_iris_clean_seed0", data_split_seed=0,
                 best_genome=_genome(["quantum", "conv", "attn", "conv"]))
    write_result(tmp_path, "quantum_iris_clean_seed1", status="failed",
                 data_split_seed=1, error="boom", best_genome=None)
    records, _, _ = agg.load_results(tmp_path)
    q = agg.collect(records, {"iris"}, {"clean"}, [0, 1])[
        ("iris", "clean", "EvoTS (quantum)")]["quantum"]
    assert q["n"] == 1 and q["mean"] == pytest.approx(1.0)


# -------------------------------------------------------------- rendering

def test_blank_for_baselines_and_exaqc(tmp_path):
    write_result(tmp_path, "classical_iris_clean_seed0", condition="classical",
                 best_genome=_genome(["attn", "conv", "attn", "conv"]))
    write_result(tmp_path, "baseline-logreg_iris_clean_seed0",
                 model_family="classical_baseline", model="logreg",
                 best_genome=None)

    agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
              "--split-modes", "clean", "--seeds", "0"])
    md = (tmp_path / "comparison.md").read_text()

    assert "Quantum blocks" in md
    rows = [l for l in md.splitlines() if l.startswith("| clean |")]
    logreg = [l for l in rows if "logreg" in l][0]
    assert "| -- |" in logreg, f"baseline row should have a blank count: {logreg}"
    exaqc = [l for l in md.splitlines() if agg.EXAQC_LABEL in l and l.startswith("|")][0]
    assert "| -- |" in exaqc, f"EXAQC row should have a blank count: {exaqc}"


def test_csv_carries_both_numbers(tmp_path):
    write_result(tmp_path, "quantum_iris_clean_seed0", data_split_seed=0,
                 best_genome=_genome(["attn", "quantum", "conv", "quantum"]))
    agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
              "--split-modes", "clean", "--seeds", "0"])
    rows = list(csv.DictReader(open(tmp_path / "comparison.csv")))
    q = [r for r in rows if r["system"] == "EvoTS (quantum)"][0]
    assert float(q["quantum_blocks_mean"]) == pytest.approx(2.0)
    assert float(q["quantum_blocks_zero_fraction"]) == pytest.approx(0.0)
    base = [r for r in rows if r["system"] == agg.EXAQC_LABEL][0]
    assert base["quantum_blocks_mean"] == ""


def test_markdown_explains_how_to_read_the_column(tmp_path):
    write_result(tmp_path, "quantum_iris_clean_seed0", data_split_seed=0,
                 best_genome=_genome(["attn", "conv", "attn", "conv"]))
    agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
              "--split-modes", "clean", "--seeds", "0"])
    md = (tmp_path / "comparison.md").read_text()
    assert "free to reject quantum" in md
    assert "classical architectures" in md
