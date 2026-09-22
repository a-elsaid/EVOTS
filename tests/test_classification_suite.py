"""
The suite runner has to survive a night alone on a cluster.

The behaviours that matter: it resumes without redoing finished work, it records
a failure instead of dying on it, it never lets a --set silently change which run
a result belongs to, and when the data is missing it says exactly which file to
copy where rather than failing deep inside a training loop.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import run_classification_suite as suite  # noqa: E402


# ------------------------------------------------------------------ naming

def test_run_name_encodes_condition_dataset_split_and_seed():
    assert suite.run_name("classical", "iris", "exaqc", 7) == "classical_iris_exaqc_seed7"
    assert suite.run_name("quantum", "iris", "exaqc", 7) == "quantum_iris_exaqc_seed7"


def test_run_names_are_unique_across_the_whole_sweep():
    names = {suite.run_name(c, d, m, s)
             for c in ("classical", "quantum")
             for d in suite.ALL_DATASETS
             for m in suite.ALL_SPLIT_MODES
             for s in suite.DEFAULT_SEEDS}
    assert len(names) == 2 * 4 * 2 * 10, "two runs would share an output directory"


def test_defaults_are_four_datasets_two_modes_ten_seeds():
    args = suite.parse_args([])
    assert sorted(args.datasets) == sorted(suite.ALL_DATASETS)
    assert sorted(args.split_modes) == sorted(suite.ALL_SPLIT_MODES)
    assert args.seeds == list(range(10))


# ------------------------------------------------------------------ resume

def test_already_ok_only_for_status_ok(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"status": "ok"}))
    assert suite.already_ok(p) is True

    p.write_text(json.dumps({"status": "failed"}))
    assert suite.already_ok(p) is False


def test_already_ok_is_false_for_missing_or_corrupt(tmp_path):
    """A half-written file must mean "rerun", never "done"."""
    assert suite.already_ok(tmp_path / "nope.json") is False
    bad = tmp_path / "bad.json"
    bad.write_text('{"status": "ok"')     # truncated mid-write
    assert suite.already_ok(bad) is False


# ------------------------------------------------------------------ command

def _sets_from(cmd):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--set"]


def test_command_sets_identity_overrides():
    cmd = suite.build_command("wine", "exaqc", 5, Path("/out"), [])
    sets = _sets_from(cmd)
    assert "run.name=classical_wine_exaqc_seed5" in sets
    assert "run.condition=classical" in sets
    assert "evo.random_seed=5" in sets
    assert "data.tabular.random_seed=5" in sets
    assert "data.tabular.split_mode=exaqc" in sets
    assert "run.logs_dir=/out" in sets
    assert "run.checkpoints_dir=/out/classical_wine_exaqc_seed5/checkpoints" in sets
    assert str(suite.CONFIG_DIR / "wine.yml") in cmd


def test_checkpoint_dirs_are_unique_per_run():
    """The config's shared checkpoints/ would have runs overwriting each other."""
    dirs = set()
    for d in suite.ALL_DATASETS:
        for m in suite.ALL_SPLIT_MODES:
            for seed in (0, 1):
                sets = _sets_from(suite.build_command(d, m, seed, Path("/out"), []))
                ckpt = [x for x in sets if x.startswith("run.checkpoints_dir=")]
                assert len(ckpt) == 1
                dirs.add(ckpt[0])
    assert len(dirs) == len(suite.ALL_DATASETS) * len(suite.ALL_SPLIT_MODES) * 2


def test_user_overrides_come_first_so_suite_keys_win():
    """
    load_cfg applies overrides in order, last wins. A user --set that collides
    with a sweep key must not be able to mislabel a run.
    """
    cmd = suite.build_command("iris", "clean", 3, Path("/out"),
                              ["eval.device=cpu", "evo.random_seed=999",
                               "run.checkpoints_dir=/shared/ckpt"])
    sets = _sets_from(cmd)
    assert "eval.device=cpu" in sets
    assert sets.index("evo.random_seed=999") < sets.index("evo.random_seed=3")
    assert (sets.index("run.checkpoints_dir=/shared/ckpt")
            < sets.index("run.checkpoints_dir=/out/classical_iris_clean_seed3/checkpoints"))
    assert sets[-1].startswith("data.tabular.split_mode=")


def test_owned_keys_cover_every_override_the_suite_sets():
    """A suite-set key missing from OWNED_KEYS would collide without a NOTE."""
    sets = _sets_from(suite.build_command("iris", "clean", 0, Path("/out"), []))
    keys = {s.split("=", 1)[0] for s in sets}
    assert keys == suite.OWNED_KEYS


def test_passthrough_sets_are_forwarded():
    cmd = suite.build_command("iris", "clean", 0, Path("/out"),
                              ["eval.device=cpu", "evo.max_evals=3"])
    sets = _sets_from(cmd)
    assert "eval.device=cpu" in sets and "evo.max_evals=3" in sets


# ------------------------------------------------------------ crash record

def test_record_crash_writes_a_visible_failed_result(tmp_path):
    """A natively killed run must not look like one that never started."""
    suite.record_crash(tmp_path, "seeds", "clean", 4, -9,
                       tmp_path / "run.log", 12.5)
    rec = json.loads((tmp_path / "classical_seeds_clean_seed4" / "results.json").read_text())
    assert rec["status"] == "failed"
    assert rec["run_name"] == "classical_seeds_clean_seed4"
    assert rec["condition"] == "classical"
    assert rec["evo_random_seed"] == 4
    assert rec["split_mode"] == "clean"
    assert "without writing" in rec["error"]
    assert rec["recorded_by"] == "run_classification_suite"


# ------------------------------------------------------------- missing data

def test_missing_data_stops_with_a_copyable_path(monkeypatch, tmp_path, capsys):
    """When the fetch cannot supply the file, say which file and where."""
    monkeypatch.setattr(suite, "DATA_DIR", tmp_path / "tabular")

    def fetch_fails(*a, **k):
        raise RuntimeError("no network on compute node")
    monkeypatch.setattr(suite.subprocess, "run", fetch_fails)

    with pytest.raises(SystemExit) as exc:
        suite.ensure_datasets(["seeds"])
    assert exc.value.code == 2

    err = capsys.readouterr().err
    assert "seeds.csv" in err
    assert str(tmp_path / "tabular") in err
    assert "fetch_datasets.py" in err


def test_present_data_needs_no_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "DATA_DIR", tmp_path)
    (tmp_path / "iris.csv").write_text("a,target\n1,0\n")

    def should_not_run(*a, **k):
        raise AssertionError("fetch ran although the CSV was present")
    monkeypatch.setattr(suite.subprocess, "run", should_not_run)

    suite.ensure_datasets(["iris"])


def test_fetch_is_attempted_once_then_rechecked(monkeypatch, tmp_path, capsys):
    """A partial fetch (3 of 4 offline, seeds needs network) still stops."""
    data = tmp_path / "tabular"
    monkeypatch.setattr(suite, "DATA_DIR", data)
    calls = []

    def fake_fetch(*a, **k):
        calls.append(a)
        data.mkdir(parents=True, exist_ok=True)
        (data / "iris.csv").write_text("a,target\n1,0\n")   # seeds still absent
        class P: returncode, stdout, stderr = 1, "", "HTTPError"
        return P()
    monkeypatch.setattr(suite.subprocess, "run", fake_fetch)

    with pytest.raises(SystemExit):
        suite.ensure_datasets(["iris", "seeds"])
    assert len(calls) == 1, "fetch should be attempted once, not per dataset"
    assert "seeds.csv" in capsys.readouterr().err


# ------------------------------------------------------------------ dry run

def test_dry_run_launches_nothing(monkeypatch, tmp_path, capsys):
    def should_not_run(*a, **k):
        raise AssertionError("dry run launched a subprocess")
    monkeypatch.setattr(suite.subprocess, "run", should_not_run)

    code = suite.main(["--dry-run", "--datasets", "iris", "--seeds", "0", "1",
                       "--split-modes", "clean", "--out-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    assert "classical_iris_clean_seed0" in out and "classical_iris_clean_seed1" in out


def test_dry_run_marks_finished_runs_as_skip(tmp_path, capsys):
    d = tmp_path / "classical_iris_clean_seed0"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"status": "ok"}))

    suite.main(["--dry-run", "--datasets", "iris", "--seeds", "0",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])
    assert "skip (ok)" in capsys.readouterr().out


# ------------------------------------------------------------------ summary

def test_exit_code_is_nonzero_when_a_run_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)

    class Failed:
        returncode = 1
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: Failed())

    code = suite.main(["--datasets", "iris", "--seeds", "0", "--split-modes",
                       "clean", "--out-dir", str(tmp_path)])
    assert code == 1
    rec = json.loads((tmp_path / "classical_iris_clean_seed0" / "results.json").read_text())
    assert rec["status"] == "failed"


def test_one_failure_does_not_stop_the_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    seen = []

    class Failed:
        returncode = 1

    def fake_run(cmd, **k):
        seen.append(cmd)
        return Failed()
    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    suite.main(["--datasets", "iris", "--seeds", "0", "1", "2",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])
    assert len(seen) == 3, "the sweep stopped at the first failure"


# ----------------------------------------------------------------- archiving

def test_archive_moves_previous_attempt_aside(tmp_path):
    rdir = tmp_path / "classical_iris_clean_seed0"
    rdir.mkdir()
    (rdir / "results.json").write_text(json.dumps({"status": "failed"}))
    (rdir / "suite_run.log").write_text("old log")

    dest = suite.archive_previous_attempt(rdir)

    assert dest is not None and dest.exists()
    assert not rdir.exists(), "original directory should have been moved"
    assert (dest / "suite_run.log").read_text() == "old log"
    assert suite.is_archived_run_dir(dest)


def test_archive_of_a_missing_directory_is_a_no_op(tmp_path):
    assert suite.archive_previous_attempt(tmp_path / "never_ran") is None


def test_two_archives_in_the_same_second_do_not_collide(tmp_path):
    made = []
    for _ in range(3):
        rdir = tmp_path / "classical_iris_clean_seed0"
        rdir.mkdir()
        made.append(suite.archive_previous_attempt(rdir))
    assert len({m.name for m in made}) == 3
    assert all(suite.is_archived_run_dir(m) for m in made)


def test_is_archived_run_dir_only_matches_attempt_folders():
    assert not suite.is_archived_run_dir("iris_clean_seed0")
    assert not suite.is_archived_run_dir("iris_clean_seed0/checkpoints")
    assert suite.is_archived_run_dir("iris_clean_seed0.attempt-20260922T010203Z")
    assert suite.is_archived_run_dir("iris_clean_seed0.attempt-20260922T010203Z-2")


def test_rerun_archives_then_starts_clean(monkeypatch, tmp_path):
    """A failed run's directory is moved aside before the retry writes anything."""
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)

    rdir = tmp_path / "classical_iris_clean_seed0"
    rdir.mkdir(parents=True)
    (rdir / "results.json").write_text(json.dumps({"status": "failed"}))
    (rdir / "stale.txt").write_text("from the previous attempt")

    class Failed:
        returncode = 1
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: Failed())

    suite.main(["--datasets", "iris", "--seeds", "0", "--split-modes", "clean",
                "--out-dir", str(tmp_path)])

    archived = [p for p in tmp_path.iterdir() if suite.is_archived_run_dir(p)]
    assert len(archived) == 1, "previous attempt was not archived"
    assert (archived[0] / "stale.txt").exists()
    assert not (rdir / "stale.txt").exists(), "fresh attempt reused the old directory"
    assert json.loads((rdir / "results.json").read_text())["status"] == "failed"


def test_skipped_runs_are_not_archived(monkeypatch, tmp_path):
    """An ok run must be left exactly as it is."""
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    rdir = tmp_path / "classical_iris_clean_seed0"
    rdir.mkdir(parents=True)
    (rdir / "results.json").write_text(json.dumps({"status": "ok"}))

    def should_not_run(*a, **k):
        raise AssertionError("an ok run was re-executed")
    monkeypatch.setattr(suite.subprocess, "run", should_not_run)

    suite.main(["--datasets", "iris", "--seeds", "0", "--split-modes", "clean",
                "--out-dir", str(tmp_path)])

    assert not [p for p in tmp_path.iterdir() if suite.is_archived_run_dir(p)]
    assert (rdir / "results.json").exists()


def test_force_archives_even_an_ok_run(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    rdir = tmp_path / "classical_iris_clean_seed0"
    rdir.mkdir(parents=True)
    (rdir / "results.json").write_text(json.dumps({"status": "ok"}))

    class Failed:
        returncode = 1
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: Failed())

    suite.main(["--force", "--datasets", "iris", "--seeds", "0",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])

    archived = [p for p in tmp_path.iterdir() if suite.is_archived_run_dir(p)]
    assert len(archived) == 1
    assert json.loads((archived[0] / "results.json").read_text())["status"] == "ok"


# ------------------------------------------------------------------ condition

def test_condition_is_part_of_the_run_name_and_recorded():
    cmd = suite.build_command("iris", "clean", 0, Path("/out"), [],
                              condition="quantum")
    sets = _sets_from(cmd)
    assert "run.name=quantum_iris_clean_seed0" in sets
    assert "run.condition=quantum" in sets


def test_a_quantum_sweep_never_skips_a_classical_run(monkeypatch, tmp_path):
    """
    The failure this guards: a quantum sweep finding classical results.json
    files, calling them done, and producing an empty quantum column that looks
    complete.
    """
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)

    # A finished classical sweep.
    for seed in (0, 1):
        d = tmp_path / f"classical_iris_clean_seed{seed}"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({"status": "ok",
                                                    "condition": "classical"}))

    launched = []

    class Failed:
        returncode = 1

    def fake_run(cmd, **k):
        launched.append(cmd)
        return Failed()
    monkeypatch.setattr(suite.subprocess, "run", fake_run)

    suite.main(["--condition", "quantum", "--datasets", "iris",
                "--seeds", "0", "1", "--split-modes", "clean",
                "--out-dir", str(tmp_path)])

    assert len(launched) == 2, "quantum runs were skipped because of classical results"
    for cmd in launched:
        sets = _sets_from(cmd)
        assert "run.condition=quantum" in sets
    # And the classical results were left untouched.
    for seed in (0, 1):
        rec = json.loads((tmp_path / f"classical_iris_clean_seed{seed}"
                          / "results.json").read_text())
        assert rec["condition"] == "classical"


def test_same_condition_still_resumes(monkeypatch, tmp_path):
    """The skip must still work within one condition."""
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    d = tmp_path / "quantum_iris_clean_seed0"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"status": "ok"}))

    def should_not_run(*a, **k):
        raise AssertionError("a finished run of this condition was re-executed")
    monkeypatch.setattr(suite.subprocess, "run", should_not_run)

    suite.main(["--condition", "quantum", "--datasets", "iris", "--seeds", "0",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])


def test_condition_is_an_owned_key():
    assert "run.condition" in suite.OWNED_KEYS
    sets = _sets_from(suite.build_command("iris", "clean", 0, Path("/out"),
                                          ["run.condition=sneaky"],
                                          condition="quantum"))
    assert sets.index("run.condition=sneaky") < sets.index("run.condition=quantum")


def test_missing_config_for_a_condition_stops_early(tmp_path, capsys):
    code = suite.main(["--config-dir", str(tmp_path / "nope"), "--datasets", "iris",
                       "--seeds", "0", "--split-modes", "clean",
                       "--out-dir", str(tmp_path)])
    assert code == 2
    assert "no config for iris" in capsys.readouterr().err


def test_config_dir_is_used_for_the_config_path(tmp_path):
    cmd = suite.build_command("iris", "clean", 0, Path("/out"), [],
                              condition="quantum", config_dir=tmp_path)
    assert str(tmp_path / "iris.yml") in cmd


def test_condition_is_lowercased(tmp_path, capsys):
    """Quantum and quantum must not become two rows with half the seeds each."""
    assert suite.normalise_condition("Quantum") == "quantum"
    assert suite.normalise_condition("  CLASSICAL  ") == "classical"
    assert suite.normalise_condition("quantum-v2") == "quantum-v2"


@pytest.mark.parametrize("bad", ["quantum run", "quantum_v2", "qu4ntum!",
                                 "", "   ", "quantum/v2", "quantum.v2"])
def test_invalid_condition_labels_are_rejected(bad, capsys):
    with pytest.raises(SystemExit) as exc:
        suite.normalise_condition(bad)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not a valid label" in err
    assert "lowercase letters, digits and hyphens" in err


def test_uppercase_condition_runs_under_the_lowercased_name(monkeypatch, tmp_path):
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    launched = []

    class Failed:
        returncode = 1
    monkeypatch.setattr(suite.subprocess, "run",
                        lambda cmd, **k: (launched.append(cmd), Failed())[1])

    suite.main(["--condition", "Quantum", "--datasets", "iris", "--seeds", "0",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])

    sets = _sets_from(launched[0])
    assert "run.condition=quantum" in sets
    assert "run.name=quantum_iris_clean_seed0" in sets
    assert (tmp_path / "quantum_iris_clean_seed0").exists()


def test_uppercase_condition_resumes_a_lowercase_sweep(monkeypatch, tmp_path):
    """--condition Quantum must find the runs --condition quantum finished."""
    monkeypatch.setattr(suite, "ensure_datasets", lambda ds: None)
    d = tmp_path / "quantum_iris_clean_seed0"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"status": "ok"}))

    def should_not_run(*a, **k):
        raise AssertionError("a finished run was re-executed under a case variant")
    monkeypatch.setattr(suite.subprocess, "run", should_not_run)

    suite.main(["--condition", "QUANTUM", "--datasets", "iris", "--seeds", "0",
                "--split-modes", "clean", "--out-dir", str(tmp_path)])
