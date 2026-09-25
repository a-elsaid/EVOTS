"""
constraints.min_quantum_blocks: a floor the free conditions do not have.

The free quantum condition may reject quantum entirely -- about 28% of sampled
genomes contain no quantum block, and selection can drive that higher. The forced
condition guarantees at least one, so the two answer different questions and must
not be confused for one another.

The floor defaults to 0, so every pre-existing config is unaffected; a test pins
that, because a default that quietly forced quantum would change the classical
and forecasting searches too.
"""

import dataclasses
from pathlib import Path

import pytest
import torch
import yaml

from nas_ts.core.config_loader import build_experiment, load_cfg
from nas_ts.models.model_builder_v2 import build_model
from nas_ts.search.crossover import crossover_genome
from nas_ts.search.genome_init import random_genome
from nas_ts.search.mutation import mutate_genome
from nas_ts.search.repair import repair_genome
from nas_ts.utils.seeding import seed_everything

FREE_DIR = Path("configs/classification_quantum")
FORCED_DIR = Path("configs/classification_quantum_forced")
DATASETS = {"iris": (4, 3), "wine": (13, 3), "seeds": (7, 3), "breast_cancer": (30, 2)}


def _n_quantum(g):
    return sum(1 for st in g.stages for b in st.blocks if b.block_type == "quantum")


def _exp(directory, name):
    return build_experiment(load_cfg(str(directory / f"{name}.yml"), overrides=None))


# ------------------------------------------------------------ the default

def test_floor_defaults_to_zero():
    from nas_ts.core.config import GenomeConstraints
    assert GenomeConstraints().min_quantum_blocks == 0


@pytest.mark.parametrize("cfg_path", [
    "configs/classification/iris.yml",
    "configs/classification_quantum/iris.yml",
    "configs/master_config.yml",
])
def test_existing_configs_keep_a_zero_floor(cfg_path):
    """Nothing that existed before this change may start forcing quantum."""
    exp = build_experiment(load_cfg(cfg_path, overrides=None))
    assert exp.genome_constraints.min_quantum_blocks == 0


# ------------------------------------------------------- forced vs free

@pytest.mark.parametrize("name", sorted(DATASETS))
def test_forced_configs_always_produce_a_quantum_block(name):
    exp = _exp(FORCED_DIR, name)
    assert exp.genome_constraints.min_quantum_blocks == 1
    seed_everything(0)
    for i in range(40):
        g = random_genome(exp.search_space, exp.genome_constraints)
        assert _n_quantum(g) >= 1, f"{name}: genome {i} has no quantum block"
        assert g.quantum_block.enabled, f"{name}: genome {i} has blocks but spec disabled"


def test_free_configs_still_sometimes_produce_zero():
    """The free condition must stay free; if it never produced zero, the two
    conditions would be measuring the same thing."""
    exp = _exp(FREE_DIR, "iris")
    assert exp.genome_constraints.min_quantum_blocks == 0
    seed_everything(0)
    counts = [_n_quantum(random_genome(exp.search_space, exp.genome_constraints))
              for _ in range(60)]
    assert any(c == 0 for c in counts), "free config never produced a zero-quantum genome"
    assert any(c > 0 for c in counts), "free config never produced a quantum genome"


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_forced_genomes_build_and_run(name):
    d_in, k = DATASETS[name]
    exp = _exp(FORCED_DIR, name)
    seed_everything(1)
    for i in range(8):
        g = random_genome(exp.search_space, exp.genome_constraints)
        model = build_model(g, exp.eval_config.task, d_in=d_in, d_out=d_in,
                            num_classes=k).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, d_in))
        assert out.shape == (2, k)


def test_floor_survives_mutation_and_crossover():
    """Every operator ends in repair, so the floor must hold after each."""
    exp = _exp(FORCED_DIR, "iris")
    ss, cons, evo = exp.search_space, exp.genome_constraints, exp.evolution
    seed_everything(2)
    g = random_genome(ss, cons)
    for i in range(25):
        g = mutate_genome(g, ss, evo, cons)
        assert _n_quantum(g) >= 1, f"mutation {i} dropped the last quantum block"

    p1, p2 = random_genome(ss, cons), random_genome(ss, cons)
    for i in range(15):
        child = crossover_genome(p1, p2, ss, evo, cons)
        assert _n_quantum(child) >= 1, f"crossover {i} produced a zero-quantum child"


# ------------------------------------------- interaction with other constraints

def test_conversion_does_not_change_the_total_block_count():
    """The floor is met by converting, not inserting, so min/max_blocks hold."""
    exp = _exp(FREE_DIR, "iris")
    cons = dataclasses.replace(exp.genome_constraints, min_quantum_blocks=3)
    seed_everything(3)
    for _ in range(20):
        g = random_genome(exp.search_space, exp.genome_constraints)
        before = sum(len(st.blocks) for st in g.stages)
        g = repair_genome(g, exp.search_space, cons)
        after = sum(len(st.blocks) for st in g.stages)
        assert before == after
        assert cons.min_blocks <= after <= cons.max_blocks


def test_conversion_cannot_break_the_max_block_type_ceilings():
    """Converting a conv block to quantum only ever reduces the conv count."""
    exp = _exp(FREE_DIR, "iris")
    cons = dataclasses.replace(exp.genome_constraints, min_quantum_blocks=4)
    seed_everything(4)
    for _ in range(20):
        g = repair_genome(random_genome(exp.search_space, exp.genome_constraints),
                          exp.search_space, cons)
        conv = sum(1 for st in g.stages for b in st.blocks if b.block_type == "conv")
        assert conv <= cons.max_conv_blocks


def test_a_floor_above_the_block_budget_still_builds():
    """
    An unsatisfiable floor converts everything it can and stops, rather than
    growing the genome past max_blocks or raising. The genome must still build --
    a build failure scores inf and reads as a bad architecture.
    """
    exp = _exp(FREE_DIR, "iris")
    cons = dataclasses.replace(exp.genome_constraints, min_quantum_blocks=99)
    seed_everything(5)
    for _ in range(6):
        g = repair_genome(random_genome(exp.search_space, exp.genome_constraints),
                          exp.search_space, cons)
        blocks = [b for st in g.stages for b in st.blocks]
        assert all(b.block_type == "quantum" for b in blocks), "not every block converted"
        assert len(blocks) <= cons.max_blocks, "genome grew past max_blocks"
        model = build_model(g, exp.eval_config.task, d_in=4, d_out=4, num_classes=3).eval()
        with torch.no_grad():
            assert model(torch.randn(2, 1, 4)).shape == (2, 3)


def test_repair_without_constraints_does_not_enforce_a_floor():
    """The parameter is optional; existing two-argument callers are unchanged."""
    exp = _exp(FREE_DIR, "iris")
    seed_everything(6)
    g = random_genome(exp.search_space, exp.genome_constraints)
    for st in g.stages:
        for b in st.blocks:
            b.block_type = "attn"
    g = repair_genome(g, exp.search_space)          # no constraints passed
    assert _n_quantum(g) == 0


# ------------------------------------------------------------ the configs

def test_forced_directory_holds_the_same_four_filenames():
    assert sorted(p.name for p in FORCED_DIR.glob("*.yml")) == \
           sorted(p.name for p in FREE_DIR.glob("*.yml"))


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_forced_config_differs_only_by_the_floor_and_the_run_name(name):
    free = load_cfg(str(FREE_DIR / f"{name}.yml"), overrides=None)
    forced = load_cfg(str(FORCED_DIR / f"{name}.yml"), overrides=None)

    assert forced["run"]["name"] == free["run"]["name"] + "_forced"
    assert forced["constraints"].pop("min_quantum_blocks") == 1
    free["constraints"].pop("min_quantum_blocks", None)
    assert forced["constraints"] == free["constraints"]

    for section in ("task", "data", "eval", "selection", "evo", "search_space"):
        assert forced[section] == free[section], f"{name}: {section} differs"
