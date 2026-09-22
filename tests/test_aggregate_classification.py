"""
The comparison table must never overstate what was actually run.

The failure modes that would matter in a paper: averaging a superseded attempt
alongside the run that replaced it, silently dropping a crashed seed so a cell
reads as complete, mixing baseline numbers into the EvoTS row, or presenting
"clean" numbers next to EXAQC's published figures as if they were comparable.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import aggregate_classification as agg  # noqa: E402


def write_result(root: Path, name: str, **fields):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "run_name": name,
        "dataset_path": "data/tabular/iris.csv",
        "split_mode": "clean",
        "data_split_seed": 0,
        "test_accuracy": 0.90,
        "test_macro_accuracy": 0.90,
    }
    payload.update(fields)
    (d / "results.json").write_text(json.dumps(payload))
    return d / "results.json"


# ------------------------------------------------------------------ archived

def test_archived_attempts_are_skipped(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", test_accuracy=1.00)
    write_result(tmp_path, "iris_clean_seed0.attempt-20260922T010203Z",
                 status="failed", test_accuracy=0.10)

    records, archived, _ = agg.load_results(tmp_path)
    assert len(records) == 1
    assert len(archived) == 1

    cells = agg.collect(records, {"iris"}, {"clean"}, [0])
    cell = cells[("iris", "clean", "EvoTS (classical)")]
    assert cell["accuracy"]["n"] == 1
    assert cell["accuracy"]["mean"] == pytest.approx(100.0)
    assert cell["failed"] == [], "an archived attempt was counted as a failure"


def test_archived_ok_attempt_is_also_skipped(tmp_path):
    """--force archives an ok run; counting both would double-count the seed."""
    write_result(tmp_path, "iris_clean_seed0", test_accuracy=0.80)
    write_result(tmp_path, "iris_clean_seed0.attempt-20260922T010203Z",
                 test_accuracy=1.00)

    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean"}, [0])
    assert cells[("iris", "clean", "EvoTS (classical)")]["accuracy"]["n"] == 1
    assert cells[("iris", "clean", "EvoTS (classical)")]["accuracy"]["mean"] == pytest.approx(80.0)


# -------------------------------------------------------- failed and missing

def test_failed_runs_are_listed_and_not_averaged(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", test_accuracy=0.90)
    write_result(tmp_path, "iris_clean_seed1", status="failed",
                 data_split_seed=1, test_accuracy=None,
                 error="RuntimeError: worker died")

    records, _, _ = agg.load_results(tmp_path)
    cell = agg.collect(records, {"iris"}, {"clean"}, [0, 1])[("iris", "clean", "EvoTS (classical)")]

    assert cell["accuracy"]["n"] == 1, "a failed run was averaged in"
    assert len(cell["failed"]) == 1
    assert "worker died" in cell["failed"][0][1]


def test_missing_seeds_are_reported(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", data_split_seed=0)
    records, _, _ = agg.load_results(tmp_path)
    cell = agg.collect(records, {"iris"}, {"clean"}, [0, 1, 2])[("iris", "clean", "EvoTS (classical)")]
    assert cell["missing_seeds"] == [1, 2]


def test_unreadable_results_are_surfaced(tmp_path):
    d = tmp_path / "iris_clean_seed0"
    d.mkdir(parents=True)
    (d / "results.json").write_text("{not json")
    records, _, unreadable = agg.load_results(tmp_path)
    assert records == [] and len(unreadable) == 1


def test_problems_appear_in_the_markdown(tmp_path, capsys):
    write_result(tmp_path, "iris_clean_seed0")
    write_result(tmp_path, "iris_clean_seed1", status="failed", data_split_seed=1,
                 error="BrokenProcessPool")

    agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
              "--split-modes", "clean", "--seeds", "0", "1", "2"])
    md = (tmp_path / "comparison.md").read_text()
    assert "Runs not included" in md
    assert "BrokenProcessPool" in md
    assert "missing" in md and "2" in md


# ------------------------------------------------------- systems and labels

def test_baselines_are_separated_from_evots_by_model_family(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", test_accuracy=0.70)
    write_result(tmp_path, "baseline-logreg_iris_clean_seed0",
                 model_family="classical_baseline", model="logreg",
                 test_accuracy=1.00)

    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean"}, [0])

    assert cells[("iris", "clean", "EvoTS (classical)")]["accuracy"]["mean"] == pytest.approx(70.0)
    assert cells[("iris", "clean", "logreg")]["accuracy"]["mean"] == pytest.approx(100.0)


def test_system_of_uses_model_family_not_the_run_name():
    assert agg.system_of({"run_name": "baseline-logreg_x"}) == "EvoTS (classical)"
    assert agg.system_of({"model_family": "classical_baseline",
                          "model": "svc_rbf"}) == "svc_rbf"


def test_exaqc_mode_baselines_are_labelled_unselected(tmp_path):
    write_result(tmp_path, "baseline-logreg_iris_exaqc_seed0", split_mode="exaqc",
                 model_family="classical_baseline", model="logreg")
    write_result(tmp_path, "baseline-logreg_iris_clean_seed0", split_mode="clean",
                 model_family="classical_baseline", model="logreg")
    write_result(tmp_path, "iris_exaqc_seed0", split_mode="exaqc")

    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean", "exaqc"}, [0])
    rows = agg.build_rows(cells, ["iris"], ["clean", "exaqc"])

    by = {(r["system"], r["split_mode"]): r for r in rows}
    assert by[("logreg", "exaqc")]["note"].strip() == "[no tuning]"
    assert by[("logreg", "clean")]["note"] == ""
    assert by[("EvoTS (classical)", "exaqc")]["note"] == "", \
        "EvoTS does select in exaqc mode"


def test_only_exaqc_rows_are_marked_comparable(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", split_mode="clean")
    write_result(tmp_path, "iris_exaqc_seed0", split_mode="exaqc")
    records, _, _ = agg.load_results(tmp_path)
    rows = agg.build_rows(agg.collect(records, {"iris"}, {"clean", "exaqc"}, [0]),
                          ["iris"], ["clean", "exaqc"])
    for r in rows:
        assert r["comparable_to_exaqc"] == (r["split_mode"].startswith("exaqc"))


# ------------------------------------------------------------------- exaqc

@pytest.mark.parametrize("dataset,n,mean,best,worst", [
    ("iris", 5, 80.0, 90.0, 70.0),
    ("seeds", 5, 92.88, 95.2, 90.5),
    ("wine", 6, 82.33, 91.2, 75.0),
    ("breast_cancer", 4, 89.9, 91.0, 88.6),
])
def test_published_exaqc_numbers(dataset, n, mean, best, worst):
    cell = agg.exaqc_cell(dataset)
    assert cell["accuracy"]["n"] == n
    assert cell["accuracy"]["mean"] == pytest.approx(mean, abs=0.02)
    assert cell["accuracy"]["best"] == pytest.approx(best)
    assert cell["accuracy"]["worst"] == pytest.approx(worst)


def test_exaqc_row_is_present_and_labelled(tmp_path):
    write_result(tmp_path, "iris_clean_seed0")
    records, _, _ = agg.load_results(tmp_path)
    rows = agg.build_rows(agg.collect(records, {"iris"}, {"clean"}, [0]),
                          ["iris"], ["clean"])
    exaqc = [r for r in rows if r["system"] == agg.EXAQC_LABEL]
    assert len(exaqc) == 1
    assert "val = test" in exaqc[0]["system"]


# ------------------------------------------------------------------ outputs

def test_writes_csv_markdown_and_latex(tmp_path):
    write_result(tmp_path, "iris_clean_seed0")
    code = agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
                     "--split-modes", "clean", "--seeds", "0"])
    assert code == 0
    for name in ("comparison.csv", "comparison.md", "comparison.tex"):
        assert (tmp_path / name).exists(), f"{name} not written"

    with open(tmp_path / "comparison.csv") as f:
        rows = list(csv.DictReader(f))
    assert any(r["system"] == "EvoTS (classical)" for r in rows)
    assert any(r["system"] == agg.EXAQC_LABEL for r in rows)
    assert "n_failed" in rows[0] and "missing_seeds" in rows[0]

    tex = (tmp_path / "comparison.tex").read_text()
    assert r"\begin{tabular}" in tex and r"\end{tabular}" in tex
    # Underscores must be escaped in the table body. Comment lines (%) are not
    # typeset, so they are exempt.
    body = [l for l in tex.splitlines() if not l.startswith("%")]
    assert "_" not in "\n".join(body).replace(r"\_", ""), "unescaped underscore in LaTeX"


def test_out_dir_can_differ_from_results_dir(tmp_path):
    results = tmp_path / "res"
    out = tmp_path / "report"
    write_result(results, "iris_clean_seed0")
    agg.main(["--results-dir", str(results), "--out-dir", str(out),
              "--datasets", "iris", "--split-modes", "clean", "--seeds", "0"])
    assert (out / "comparison.md").exists()


def test_missing_results_dir_exits_nonzero(tmp_path):
    assert agg.main(["--results-dir", str(tmp_path / "nope")]) == 2


# ------------------------------------------------------------------ condition

def test_conditions_become_separate_rows(tmp_path):
    write_result(tmp_path, "classical_iris_clean_seed0", condition="classical",
                 test_accuracy=0.80)
    write_result(tmp_path, "quantum_iris_clean_seed0", condition="quantum",
                 test_accuracy=0.60)

    records, _, _ = agg.load_results(tmp_path)
    cells = agg.collect(records, {"iris"}, {"clean"}, [0])

    assert cells[("iris", "clean", "EvoTS (classical)")]["accuracy"]["mean"] == pytest.approx(80.0)
    assert cells[("iris", "clean", "EvoTS (quantum)")]["accuracy"]["mean"] == pytest.approx(60.0)


def test_a_new_condition_appears_without_code_changes(tmp_path):
    """The label comes from the data, so an unseen condition still gets a row."""
    write_result(tmp_path, "hybrid_iris_clean_seed0", condition="hybrid-v2",
                 test_accuracy=0.75)
    records, _, _ = agg.load_results(tmp_path)
    rows = agg.build_rows(agg.collect(records, {"iris"}, {"clean"}, [0]),
                          ["iris"], ["clean"])
    assert any(r["system"] == "EvoTS (hybrid-v2)" for r in rows)


def test_records_without_condition_count_as_classical_with_a_warning(tmp_path):
    write_result(tmp_path, "iris_clean_seed0", test_accuracy=0.90)  # no condition
    records, _, _ = agg.load_results(tmp_path)

    undeclared = []
    cells = agg.collect(records, {"iris"}, {"clean"}, [0], undeclared=undeclared)

    assert cells[("iris", "clean", "EvoTS (classical)")]["accuracy"]["n"] == 1
    assert undeclared == ["iris_clean_seed0"]


def test_undeclared_condition_is_reported_in_the_markdown(tmp_path):
    write_result(tmp_path, "iris_clean_seed0")   # no condition field
    agg.main(["--results-dir", str(tmp_path), "--datasets", "iris",
              "--split-modes", "clean", "--seeds", "0"])
    md = (tmp_path / "comparison.md").read_text()
    assert "no condition recorded" in md
    assert "iris_clean_seed0" in md


def test_baselines_are_not_given_a_condition(tmp_path):
    """Baselines are keyed by model, not condition; they have no condition."""
    write_result(tmp_path, "baseline-logreg_iris_clean_seed0",
                 model_family="classical_baseline", model="logreg")
    records, _, _ = agg.load_results(tmp_path)
    undeclared = []
    cells = agg.collect(records, {"iris"}, {"clean"}, [0], undeclared=undeclared)
    assert ("iris", "clean", "logreg") in cells
    assert undeclared == [], "a baseline was flagged for having no condition"


def test_evots_conditions_are_ordered_before_baselines(tmp_path):
    write_result(tmp_path, "quantum_iris_clean_seed0", condition="quantum")
    write_result(tmp_path, "baseline-logreg_iris_clean_seed0",
                 model_family="classical_baseline", model="logreg")
    write_result(tmp_path, "classical_iris_clean_seed0", condition="classical")

    records, _, _ = agg.load_results(tmp_path)
    rows = agg.build_rows(agg.collect(records, {"iris"}, {"clean"}, [0]),
                          ["iris"], ["clean"])
    systems = [r["system"] for r in rows if r["system"] != agg.EXAQC_LABEL]
    assert systems == ["EvoTS (classical)", "EvoTS (quantum)", "logreg"]
