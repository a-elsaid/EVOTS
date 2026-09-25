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

If that prints a table, the classical pipeline works. The numbers are meaningless
at this budget — 4 genomes, 3 epochs.

### ... and one for the quantum conditions

Worth running before committing cluster time: it is the only check that
`tensorcircuit` imports, that circuits compile, and that the quantum configs are
where the suite expects them. Takes a couple of minutes, mostly circuit compilation.

```bash
PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --condition quantum --config-dir configs/classification_quantum \
  --datasets iris --seeds 0 --split-modes clean --out-dir logs/smoke \
  --set eval.device=cpu --set evo.num_workers=1 \
  --set evo.max_evals=4 --set eval.training_steps=3 \
  --set evo.population_size=4 --set eval.extra_training_steps=50

PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --condition quantum-forced --config-dir configs/classification_quantum_forced \
  --datasets iris --seeds 0 --split-modes clean --out-dir logs/smoke \
  --set eval.device=cpu --set evo.num_workers=1 \
  --set evo.max_evals=4 --set eval.training_steps=3 \
  --set evo.population_size=4 --set eval.extra_training_steps=50

PYTHONPATH=$PWD python tools/aggregate_classification.py \
  --results-dir logs/smoke --datasets iris --split-modes clean --seeds 0
cat logs/smoke/comparison.md
```

The table should now show `EvoTS (classical)`, `EvoTS (quantum)` and
`EvoTS (quantum-forced)` as separate rows, with a non-blank **Quantum blocks**
column on the two quantum rows.

## 4. The full run — three conditions

There are three EvoTS conditions. Each is a separate sweep with its own config
directory and its own `--condition` label; the label becomes part of every run
name and its own row in the comparison table.

**Run them in this order, all into the same `--out-dir`, and aggregate last:**

```bash
OUT=logs/classification

# 1. classical: no quantum blocks in the search space
PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --out-dir $OUT

# 2. quantum (free): quantum joins block_types, the search may or may not use it
PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --condition quantum --config-dir configs/classification_quantum \
  --out-dir $OUT

# 3. quantum-forced: every genome carries at least one quantum block
PYTHONPATH=$PWD python experiments/run_classification_suite.py \
  --condition quantum-forced --config-dir configs/classification_quantum_forced \
  --out-dir $OUT

# 4. baselines — ONCE, not per condition (see section 5)
PYTHONPATH=$PWD python tools/classical_baselines.py --out-dir $OUT

# 5. the table — LAST, after everything above
PYTHONPATH=$PWD python tools/aggregate_classification.py --results-dir $OUT
```

Each sweep is 4 datasets x 2 split modes x 10 seeds = **80 runs**, each a full
EXAQC-matched search (500 genomes, population 50, 200 epochs per genome), so all
three conditions together are 240 runs. Every run is a separate subprocess, so one
crash kills one run, not the sweep.

Three things that are easy to get wrong:

- **One `--out-dir` for everything.** Run names are condition-prefixed
  (`classical_iris_clean_seed0`, `quantum_iris_clean_seed0`, ...) so they cannot
  collide, and the aggregation needs every condition and the baselines in one
  directory to build a single table. Separate directories give you three
  single-row tables instead of one comparison.
- **Aggregate last.** It reads whatever is present; run it early and rows are
  simply missing. It is cheap and re-runnable, so run it again whenever more
  results land.
- **Conditions are independent sweeps.** You can run them on different days, or
  only some of them; the table shows whatever exists and lists the rest under
  *Runs not included*.

Exit code is non-zero if any run failed, so a scheduler will notice.

Under a job scheduler, wrap each condition in your usual submission script and give
it plenty of walltime. If a job is killed, just submit it again — see resuming below.

### What the three conditions answer

| Condition | Config directory | Question it answers |
|---|---|---|
| `classical` | `configs/classification` | How good is a searched transformer here? |
| `quantum` | `configs/classification_quantum` | Is a quantum circuit useful **when the search may decline it**? |
| `quantum-forced` | `configs/classification_quantum_forced` | What is the best **circuit-containing** architecture? |

The free `quantum` condition does not guarantee a quantum block: `block_types` has
four entries, so about 28% of freshly sampled genomes contain none, and selection
can push that higher if circuits do not pay for themselves. See section 9.

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

## 5. Classical baselines — run once, not per condition

Logistic regression, SVC (RBF) and a small MLP, on **exactly** the same splits —
they are built by calling EvoTS's own data loader, not a reimplementation.

**These are condition-independent.** They do not use the EvoTS search at all, so
they are identical whichever conditions you ran; the aggregation keys them by model
(`model_family: classical_baseline`), not by condition. Running them again after
each sweep just recomputes the same numbers, and re-running them is harmless but
pointless.

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

Columns: accuracy and macro (mean-class) accuracy as `mean ± std (best, worst, n)`,
then **Quantum blocks** — the mean number of quantum blocks in each run's winning
genome with the fraction of winners that had none, blank for baselines and for
EXAQC. Read that column before the accuracies on any quantum row; section 9
explains why.

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
parallel by default (matching EXAQC). **Quantum conditions are far slower than classical.** On the two-minute smoke
above, the same budget took ~8x longer for the quantum condition than the
classical one (61 s vs 8 s), and the forced condition longer still, because every
genome carries a circuit. At real budget the multiplier depends on how many
quantum blocks each genome ends up with:

| quantum blocks in a genome | min/evaluation | peak GB |
|---|---|---|
| 0 (classical) | 25.6 | 0.6 |
| 1 | 42.6 | 1.8 |
| 2 | 63.6 | 1.9 |
| 4 | 99.8 | 2.0 |

Measured on CPU at breast-cancer scale with amplitude encoding at 12 qubits. Cost
grows roughly linearly with the number of quantum blocks; peak memory plateaus,
because every block in a genome shares one compiled circuit. Budget the quantum
conditions at **2-4x the classical wall clock**, and the forced condition at the
upper end of that.

**Run one cell first and time it**, then
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

### Each condition's configs live in their own directory, under the same four filenames

The suite looks up `<config-dir>/<dataset>.yml`, so every condition needs the same
four filenames in a different directory — never `iris_quantum.yml`:

```
configs/classification/{iris,wine,seeds,breast_cancer}.yml                  # classical
configs/classification_quantum/{iris,wine,seeds,breast_cancer}.yml          # quantum
configs/classification_quantum_forced/{iris,wine,seeds,breast_cancer}.yml   # quantum-forced
```

The suite stops with exit 2 if a config is missing from the directory you point it
at, rather than failing 80 times in a row.

### In the free quantum condition, the search may reject quantum entirely

`block_types` in the quantum configs is `["attn", "inv_attn", "conv", "quantum"]`,
so each block is quantum with probability about 1/4 and **roughly 28% of freshly
sampled genomes contain no quantum block at all**. Nothing requires one. Selection
can push that fraction either way, and since a quantum block costs 2-4x a classical
one for no guaranteed accuracy gain, it may well push it toward 100%.

That is a legitimate result — "the search declined to use circuits" is worth
knowing — but it changes how the row must be read. **A quantum row whose winners
contain no quantum blocks is reporting classical architectures that happened to be
found under a quantum budget.** It is not evidence about quantum circuits either
way.

The comparison table makes this visible: the **Quantum blocks** column gives the
mean number of quantum blocks in each run's winning genome, with the fraction of
runs whose winner had none.

```
| clean | EvoTS (classical)       | ... | 0.0 (100% none) |
| clean | EvoTS (quantum)         | ... | 2.0 (0% none)   |
| clean | EvoTS (quantum-forced)  | ... | 1.5 (0% none)   |
| clean | logreg                  | ... | --              |
```

Read it before reading the accuracies:

- **`0.0 (100% none)`** on a quantum row — the search rejected quantum. Compare
  that row against `classical`, not against EXAQC's circuits.
- **a high `none` fraction, say 60%** — the row mixes quantum and classical
  winners, and its mean accuracy is an average over two different model families.
- **`(0% none)`** — every winner used a circuit; the row means what it appears to.

`quantum-forced` exists for exactly this reason: `constraints.min_quantum_blocks: 1`
guarantees every genome carries a circuit, so that row always answers "best
circuit-containing architecture" and can never quietly become a classical row. The
blank `--` on baseline and EXAQC rows means there is no genome of ours to inspect.

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

### Memory per worker, and why the qubit ranges are capped

Simulating an n-qubit circuit holds 2^n complex amplitudes **per token**, vmapped
over batch x tokens. Peak memory per worker, measured on CPU at breast-cancer scale
(960 tokens per call):

| encoding / readout | n=12 | n=14 |
|---|---|---|
| amplitude / state | 1.8 GB | 3.8 GB |
| amplitude / expval_z | 2.4 GB | 8.6 GB |
| angle / state | 1.3 GB | 2.1 GB |
| angle / expval_z | 2.5 GB | 7.7 GB |

The configs request **11 workers**, so multiply by 11 for the node: 8.6 GB per
worker is ~95 GB. That is why `quantum_amplitude_qubits_range` stops at 12, and why
breast cancer caps angle at 12 while the smaller datasets allow 14.

A worker that runs out of memory dies, and the suite records that genome as `inf`
fitness — **indistinguishable from a genuinely bad architecture**. If a quantum
sweep produces suspiciously many `inf` results, suspect memory before suspecting
the search.

**These are CPU figures and the cluster runs on GPU.** Expect the wall-clock
multipliers to shrink (the arithmetic parallelises well) but the memory ceiling to
bind *sooner*, because GPU RAM per worker is smaller than host RAM. Re-measure
before trusting either bound there:

```bash
PYTHONPATH=$PWD python tools/bench_quantum.py --device cuda --backward \
  --qubits 8,10,12,14 --batch 32 --tokens 30
```

If GPU memory is tight, the lever is the qubit range in the config, or
`evo.num_workers`, not the circuit cache.

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
