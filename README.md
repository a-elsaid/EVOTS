# EvoTS: Evolutionary Transformer Search for Time Series Forecasting

EvoTS is an evolutionary neural architecture search framework for discovering task-adaptive Transformer-like models for multivariate time series forecasting. Architectures are encoded as modular genomes and evolved using a steady-state evolutionary algorithm, automatically discovering hybrid compositions of attention, convolutional, and projection components without manual design.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- pandas, numpy, matplotlib, loguru, rich, pyyaml

Install dependencies:

```bash
pip install -e .
```

## Usage

Run an experiment with a config file:

```bash
python experiments/run_exp.py --config configs/config.yml
```

Override config values at the command line:

```bash
python experiments/run_exp.py --config configs/config.yml \
  --set evo.max_evals=500 \
  --set eval.training_steps=30 \
  --set data.csv.paths='["./data/ETT/ETTh1.csv"]'
```

Print the resolved config without running:

```bash
python experiments/run_exp.py --config configs/config.yml --print-config
```

See all available config options:

```bash
python experiments/run_exp.py --help-set
```

## Configuration

The YAML config controls all aspects of the experiment. Key sections:

- `run` — experiment name and output paths
- `task` — input/output lengths, number of variables, forecasting horizon
- `data.csv` — dataset paths, train/val split ratios, normalization
- `eval` — optimizer, learning rate, training steps, early stopping
- `search_space` — tokenizer types, block types, model dimension range, depth range
- `evo` — population size, max evaluations, mutation/crossover rates, number of workers
- `constraints` — min/max total blocks per genome
- `seeds` — optional hand-crafted seed architectures (iTransformer, PatchTST)

## Results

Evaluated on ETTh1, ETTh2, ETTm1, ETTm2 under multivariate-to-multivariate forecasting (M-M). MSE comparison against iTransformer:

| Dataset | H=96 | H=192 | H=336 | H=720 |
|---------|------|-------|-------|-------|
| ETTh1 | **0.3224** vs 0.3760 | **0.3462** vs 0.4200 | **0.3527** vs 0.4200 | **0.3724** vs 0.4810 |
| ETTh2 | **0.2591** vs 0.2880 | **0.3109** vs 0.3740 | **0.3245** vs 0.4150 | **0.3485** vs 0.4200 |
| ETTm1 | **0.2580** vs 0.3290 | **0.2999** vs 0.3670 | **0.3336** vs 0.3990 | **0.3858** vs 0.4540 |
| ETTm2 | **0.1470** vs 0.1750 | **0.2000** vs 0.2410 | **0.2442** vs 0.3050 | **0.3196** vs 0.4020 |

## Citation

```bibtex
@inproceedings{evots2026,
  title     = {EvoTS: Evolutionary Transformer Search for Time Series Forecasting},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)},
  year      = {2026},
  publisher = {ACM}
}
```
