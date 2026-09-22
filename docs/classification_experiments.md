# Running the classification experiments (EvoTS vs EXAQC)

This suite answers one question: **does EvoTS beat EXAQC (arXiv 2602.03840) on Iris,
Wine, Seeds and Breast Cancer?** It runs the EvoTS search on all four datasets under
two split protocols and several seeds, runs classical baselines on exactly the same
splits, and produces a comparison table against EXAQC's published Table 1.

Everything is one command, resumable, and designed to be left alone overnight.

---

## 1. Setup (Linux)

```bash
git clone <repo> && cd EVOTS
git checkout feature/tabular-classification

python3.11 -m venv .venv
# or, with uv:  uv venv --python 3.11 .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Use Python 3.11, not whatever `python3` points at.** `torch==2.0.1` publishes no
wheels for anything newer, and `PyYAML==6.0.x` has no 3.13 wheel and fails to build
from source against current setuptools. On 3.13 the install dies partway through
with a Cython error that does not mention the Python version.

One thing about dependencies: **`requirements.txt` pins `tensorflow` and
`tensorcircuit`.** Those are only needed for the quantum work. If they fail to
install (they are large and fussy), the classical classification suite still runs —
nothing in this document imports them.

Check the install:

```bash
pip install pytest        # test-only, not in requirements.txt
PYTHONPATH=$PWD python -m pytest tests/ -q --ignore=tests/test_weight_transfer.py
```

`test_weight_transfer.py` needs `tensorcircuit`; skip it unless you installed it.
Tests that need the datasets skip cleanly if you have not fetched them yet.

## 2. Data

```bash
python fetch_datasets.py
```

Writes `data/tabular/{iris,wine,seeds,breast_cancer}.csv`. Iris, Wine and Breast
Cancer come from scikit-learn and work offline. **Seeds is downloaded from UCI and
needs network access.** Compute nodes often have none, so run this on a login node,
or fetch on any machine with network and copy the file across:

```bash
scp seeds.csv <cluster>:/path/to/EVOTS/data/tabular/
```

The suite fetches automatically if a CSV is missing, and stops with the exact path
to copy if it cannot.

## 3. Two-minute smoke test — run this first

Tiny budget, real code path. About 20 seconds on a laptop.

```bash
PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --datasets iris --seeds 0 --split-modes clean exaqc \
  --out-dir logs/smoke \
  --set eval.device=cpu --set evo.num_workers=1 \
  --set evo.max_evals=4 --set eval.training_steps=3 \
  --set evo.population_size=4 --set eval.extra_training_steps=50

PYTHONPATH=$PWD python tools/classical_baselines.py \
  --datasets iris --seeds 0 --split-modes clean exaqc --out-dir logs/smoke

PYTHONPATH=$PWD python tools/aggregate_classification.py \
  --results-dir logs/smoke --datasets iris --seeds 0

cat logs/smoke/comparison.md
```

If that prints a table, the pipeline works. The numbers are meaningless at this
budget — 4 genomes, 3 epochs.

## 4. The full run

```bash
PYTHONPATH=$PWD python experiments/run_classification_suite.py
```

That is 4 datasets x 2 split modes x 10 seeds = **80 runs**, each a full
EXAQC-matched search (500 genomes, population 50, 200 epochs per genome). Every run
is a separate subprocess, so one crash kills one run, not the sweep.

Exit code is non-zero if any run failed, so a scheduler will notice.

Under a job scheduler, wrap it in your usual submission script and give it plenty of
walltime. If the job is killed, just submit it again — see resuming below.

### Subsets

```bash
# one dataset, three seeds
python experiments/run_classification_suite.py --datasets iris --seeds 0 1 2

# only the EXAQC-comparable protocol
python experiments/run_classification_suite.py --split-modes exaqc

# see the plan without running anything
python experiments/run_classification_suite.py --dry-run
```

### Resuming

Re-run the same command. Any run whose `results.json` says `status: ok` is skipped;
failed and unfinished runs are re-executed. This is the intended way to recover from
a walltime kill — there is no separate resume flag.

Before a re-execution, the previous attempt's directory is moved aside to
`<run_name>.attempt-<UTC timestamp>` so the new attempt starts clean and the old one
survives as evidence. The aggregation ignores those folders.

To force a completed run to re-execute, add `--force`.

## 5. Classical baselines

Logistic regression, SVC (RBF) and a small MLP, on **exactly** the same splits —
they are built by calling EvoTS's own data loader, not a reimplementation.

```bash
PYTHONPATH=$PWD python tools/classical_baselines.py
```

Fast (minutes, not hours). Hyperparameters are chosen on validation in `clean` mode
and not tuned at all in `exaqc` mode, because there validation *is* the test set.

## 6. The comparison table

```bash
PYTHONPATH=$PWD python tools/aggregate_classification.py \
  --results-dir logs/classification
```

Writes three files next to the results (or wherever `--out-dir` points):

| File | Use |
|---|---|
| `comparison.md` | read it |
| `comparison.csv` | further analysis |
| `comparison.tex` | paste into the paper |

Each cell is `mean ± std (best, worst, n)` for accuracy and macro (mean-class)
accuracy, next to EXAQC's published numbers. **Failed and missing runs are listed
explicitly** under "Runs not included" — check that section before believing a row.

## 7. Where things land

```
logs/classification/
  classical_iris_clean_seed0/          one run
    results.json                       the machine-readable result
    suite_run.log                      stdout/stderr of the run
    runtime.log                        the framework's own log
    *_evaluations.csv, *_generations.csv
    *__best_finetuned.pt               the winning model
    checkpoints/                       per-run, not shared
    plots/
  classical_iris_clean_seed0.attempt-20260922T145818Z/   superseded attempt
  baseline-logreg_iris_clean_seed0/
  comparison.md / .csv / .tex
```

`results.json` is the file everything downstream reads: accuracy, macro accuracy,
loss, the winning genome, parameter count, split sizes, both seeds, the split mode,
the git SHA and whether the tree was dirty.

## 8. Runtime

**I could not measure this.** The suite was developed and smoke-tested on a laptop
with tiny budgets (3-4 genomes, 2-3 epochs); a real run is 500 genomes at 200 epochs
each, which is four orders of magnitude more compute, on different hardware.

What I can say: the four datasets are small (150-569 rows), each genome trains in
seconds to low minutes on CPU at this scale, and the search runs 11 workers in
parallel by default (matching EXAQC). **Run one cell first and time it**, then
multiply by 80:

```bash
time PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --datasets iris --seeds 0 --split-modes clean
```

---

## 9. Things you need to know

### Condition labels are lowercase with hyphens

EvoTS will be run in more than one condition (`classical` now, `quantum` later).
The label goes into every run name and becomes the row key in the comparison table.

- Valid: `classical`, `quantum`, `quantum-v2`
- **Not** valid: `quantum_v2` (underscore is the run-name separator), `Quantum v2`

`--condition Quantum` is silently lowercased to `quantum`, so it resumes the same
sweep rather than starting a parallel one. Anything outside `[a-z0-9-]+` is rejected
with exit code 2 rather than quietly creating a second set of rows.

### Quantum configs go in their own directory, with the same four filenames

The suite looks up `<config-dir>/<dataset>.yml`. So a quantum sweep needs:

```
configs/classification_quantum/{iris,wine,seeds,breast_cancer}.yml
```

— the same four filenames, in a different directory. Not `iris_quantum.yml`.

```bash
python experiments/run_classification_suite.py \
  --condition quantum --config-dir configs/classification_quantum
```

The suite stops with exit 2 if a config is missing from that directory, rather than
failing 80 times in a row.

### A whole run is bit-reproducible only at `num_workers=1`

- **Individual genomes reproduce at any worker count.** Evaluation is seeded from
  the genome's own ID, so a given genome trains identically whichever worker
  happens to run it.
- **A whole run reproduces only at `num_workers=1`.** The search is asynchronous
  and steady-state: completion order decides which parents are in the population
  when a child is bred, so with 11 workers the architectures explored differ between
  runs of the same seed. No amount of seeding fixes that; it is a property of the
  search, not the RNG.

For the experiments this is fine — the seeds give you a distribution, not one
canonical run. If you need an exactly reproducible run for debugging, add
`--set evo.num_workers=1` and expect it to be much slower.

### Keep seeds below about 2000

Per-genome seeds are derived as `base_seed * 1_000_000 + genome_id + 1`, kept inside
a 2^31-1 modulus. That is collision-free for base seeds up to **2146**; above that
the arithmetic wraps and two different runs can draw the same RNG stream. The
default seeds are 0-9, so this only matters if you pick large seed values by hand.

### `exaqc`-mode test accuracy is not a held-out estimate

The suite runs two split protocols:

| Mode | Split | Scaler | Test set |
|---|---|---|---|
| `clean` | stratified 70/15/15 | fitted on train only | held out, evaluated once |
| `exaqc` | stratified 80/20 | MinMax x pi, fitted on the **full** dataset | **same samples as validation** |

`exaqc` reproduces EXAQC's published protocol so the comparison is like-for-like.
In that mode the reported test accuracy is **not a generalisation estimate**, but
the two systems are compromised to different degrees:

- **EvoTS** is affected by both problems. Its search selects architectures, and its
  early stopping selects weights, on the very samples it then reports — and the
  scaler saw those samples' range before training. Its `exaqc` numbers are fitted,
  not held out.
- **The classical baselines** are affected only by the scaler. They do no
  hyperparameter selection at all in this mode, precisely because validation is the
  test set; each uses its default hyperparameters. Their `exaqc` numbers are
  conservative relative to EvoTS's, which is why the comparison table marks them
  `[no tuning]`.

EXAQC's own published figures carry the same selection problem EvoTS does, since
their search selects on that holdout too.

Use `exaqc` numbers only when comparing against their Table 1. Use `clean` numbers
for any claim about how well the model actually generalises. The comparison table
labels which rows are which, and the run log prints a warning every time `exaqc`
mode is used.

---

## 10. Troubleshooting

**A run failed.** Look at `logs/classification/<run_name>/suite_run.log` and the
`error` field in its `results.json`. Re-run the suite to retry it; finished runs are
skipped.

**`BrokenProcessPool` with no Python traceback.** Device auto-detection picked
`mps` (Apple Silicon), which is untested with the process backend under torch 2.0.1.
Add `--set eval.device=cpu`. On a Linux GPU node this should not occur; `auto`
resolves to CUDA there.

**The suite stops saying a dataset CSV is missing.** It could not fetch it — almost
always `seeds.csv` on a node without network. The message names the exact path to
copy it to. See section 2.

**A cell in the table has fewer runs than expected.** Look at "Runs not included" in
`comparison.md`. It distinguishes runs that *failed* (ran and broke; the error is
shown) from runs that are *missing* (never ran). Neither is ever silently averaged
away.

**A table row says `EvoTS (classical)` for a run you did not label.** Runs launched
by hand through `run_exp.py` have no condition recorded and are counted as
`classical`, with a warning naming them. Launch through the suite to label them.
