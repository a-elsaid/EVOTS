"""
The real-budget configs must be runnable unattended on the cluster.

Every option listed in these search spaces has to build and run at
input_length=1 -- a tabular row is a length-1 sequence, and several search-space
options are time-series operators that either collapse or misbehave there. A
config that only fails on the 300th genome, eight hours into a run, is the
failure mode this guards against.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from conftest import require_tabular
from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.models.model_builder_v2 import build_model
from nas_ts.search.genome_init import random_genome
from nas_ts.search.repair import repair_genome
from nas_ts.utils.seeding import seed_everything

CONFIG_DIR = Path("configs/classification")
# num_classes is held here rather than read from the CSV so the genome-building
# tests -- the ones that actually guard the search space -- run with no data
# present. test_d_in_matches_the_actual_csv is what checks these against reality.
EXPECTED = {
    "iris":          dict(d_in=4,  n_train=104, bs=16, k=3),
    "wine":          dict(d_in=13, n_train=124, bs=16, k=3),
    "seeds":         dict(d_in=7,  n_train=146, bs=16, k=3),
    "breast_cancer": dict(d_in=30, n_train=397, bs=32, k=2),
}
FINETUNE_EPOCHS = 200


def _cfg(name):
    return load_cfg(str(CONFIG_DIR / f"{name}.yml"), overrides=None)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_config_exists_and_loads(name):
    assert (CONFIG_DIR / f"{name}.yml").exists()
    build_experiment(_cfg(name))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_budget_matches_exaqc(name):
    cfg = _cfg(name)
    assert cfg["evo"]["max_evals"] == 500
    assert cfg["evo"]["population_size"] == 50
    assert cfg["eval"]["training_steps"] == 200
    assert cfg["eval"]["optimizer"] == "adam"
    assert cfg["eval"]["optimizer_kwargs"]["lr"] == pytest.approx(1e-3)
    assert cfg["eval"]["optimizer_kwargs"]["weight_decay"] == pytest.approx(1e-4)
    assert cfg["eval"]["early_stopping"] is True
    assert cfg["eval"]["device"] == "auto"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_task_settings_are_right_for_tabular(name):
    cfg = _cfg(name)
    assert cfg["task"]["task_type"] == "classification"
    assert cfg["task"]["input_length"] == 1
    assert cfg["task"]["use_norm"] is False, "WindowNorm zeroes features at L=1"
    assert cfg["task"]["d_in"] == EXPECTED[name]["d_in"]
    assert cfg["data"]["tabular"]["split_mode"] == "clean"
    assert cfg["data"]["tabular"]["path"] == f"data/tabular/{name}.csv"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_d_in_matches_the_actual_csv(name):
    """A wrong d_in builds a model that silently ignores or invents columns."""
    require_tabular(name)
    import pandas as pd
    df = pd.read_csv(f"data/tabular/{name}.csv")
    n_features = len([c for c in df.columns if c != "target"])
    assert _cfg(name)["task"]["d_in"] == n_features
    # Keep the EXPECTED table honest too, since other tests rely on it.
    assert EXPECTED[name]["k"] == df["target"].nunique()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_finetune_steps_are_the_documented_number_of_epochs(name):
    """
    extra_training_steps counts BATCHES while training_steps counts EPOCHS.
    The config comments state the arithmetic; this checks the arithmetic is real.
    """
    exp = EXPECTED[name]
    batches_per_epoch = math.ceil(exp["n_train"] / exp["bs"])
    expected = batches_per_epoch * FINETUNE_EPOCHS
    cfg = _cfg(name)
    assert cfg["data"]["tabular"]["batch_size"] == exp["bs"]
    assert cfg["eval"]["extra_training_steps"] == expected, (
        f"{name}: {batches_per_epoch} batches/epoch x {FINETUNE_EPOCHS} epochs "
        f"= {expected}, config says {cfg['eval']['extra_training_steps']}"
    )


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_quantum_is_not_in_the_search_space(name):
    text = (CONFIG_DIR / f"{name}.yml").read_text()
    block_types = _cfg(name)["search_space"]["block_types"]
    assert not any("quantum" in str(b).lower() for b in block_types)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_search_space_is_genuinely_searchable(name):
    """A space with one option everywhere is not a search."""
    ss = _cfg(name)["search_space"]
    assert ss["depth_range"][0] < ss["depth_range"][1]
    assert ss["model_dim_range"][0] < ss["model_dim_range"][1]
    assert ss["ff_mult_range"][0] < ss["ff_mult_range"][1]
    assert ss["dropout_range"][0] < ss["dropout_range"][1]
    assert len(ss["block_types"]) >= 2
    assert set(ss["stage_tokenizers"]) == {"var", "none"}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_random_genomes_build_and_run_at_input_length_one(name):
    """
    The load-bearing test: sample the space and actually forward a batch.
    Any option that cannot survive L=1 shows up here rather than mid-run.
    """
    cfg = _cfg(name)
    exp_cfg = build_experiment(cfg)
    d_in = cfg["task"]["d_in"]
    num_classes = EXPECTED[name]["k"]

    seed_everything(0)
    for i in range(40):
        genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
        model = build_model(genome, exp_cfg.eval_config.task,
                            d_in=d_in, d_out=d_in, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1, d_in))
        assert out.shape == (4, num_classes), f"genome {i} gave {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"genome {i} produced non-finite logits"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_mutated_genomes_also_stay_valid(name):
    """
    The search mutates for 500 genomes; repair must keep them buildable, not just
    the freshly sampled ones.
    """
    from nas_ts.search.mutation import mutate_genome

    cfg = _cfg(name)
    exp_cfg = build_experiment(cfg)
    d_in = cfg["task"]["d_in"]
    num_classes = EXPECTED[name]["k"]

    seed_everything(1)
    genome = random_genome(exp_cfg.search_space, exp_cfg.genome_constraints)
    for i in range(25):
        genome = mutate_genome(genome, exp_cfg.search_space,
                               exp_cfg.evolution, exp_cfg.genome_constraints)
        model = build_model(genome, exp_cfg.eval_config.task,
                            d_in=d_in, d_out=d_in, num_classes=num_classes).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 1, d_in))
        assert out.shape == (4, num_classes), f"after mutation {i}: {tuple(out.shape)}"
