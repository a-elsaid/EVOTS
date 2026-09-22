"""
evo.random_seed must actually reach the RNGs.

Before this, evo.random_seed was read by nobody: random, numpy and torch were all
unseeded, so two runs of one config explored different architectures and trained
from different weights. Only the data split was seeded. This applies to
forecasting as much as to classification.
"""

import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from nas_ts.utils.seeding import derive_task_seed, seed_everything


def _draws():
    return (random.random(), float(np.random.rand()), float(torch.rand(1)))


def test_same_seed_gives_same_draws():
    seed_everything(123)
    first = _draws()
    seed_everything(123)
    assert _draws() == first


def test_different_seed_gives_different_draws():
    seed_everything(123)
    first = _draws()
    seed_everything(124)
    assert _draws() != first


def test_seeds_every_library_not_just_one():
    """A seeding helper that missed one library would still pass a naive test."""
    seed_everything(7)
    a_py, a_np, a_torch = _draws()
    seed_everything(7)
    b_py, b_np, b_torch = _draws()
    assert a_py == b_py, "random not seeded"
    assert a_np == b_np, "numpy not seeded"
    assert a_torch == b_torch, "torch not seeded"


def test_task_seed_depends_only_on_base_and_id():
    """
    The point of per-task seeding: the worker that runs a genome must not change
    its seed. derive_task_seed takes no worker argument, so simulate the two
    cases a worker could have influenced -- called from anywhere, any order.
    """
    first = derive_task_seed(42, 7)
    for _ in range(5):
        assert derive_task_seed(42, 7) == first
    # A different individual on the same run gets a different seed.
    assert derive_task_seed(42, 8) != first


def test_task_seeds_are_distinct_per_individual():
    seeds = [derive_task_seed(42, i) for i in range(500)]
    assert len(set(seeds)) == 500


def test_task_seeds_do_not_collide_across_a_seed_sweep():
    """Seeds 0..9 x 500 genomes: no run may reuse another run's task seed."""
    all_seeds = [derive_task_seed(base, i)
                 for base in range(10) for i in range(500)]
    assert len(set(all_seeds)) == len(all_seeds)


def test_string_ids_are_supported_and_stable():
    a = derive_task_seed(3, "indiv-12")
    assert a == derive_task_seed(3, "indiv-12")
    assert a != derive_task_seed(3, "indiv-13")


def test_task_seed_is_stable_across_processes():
    """
    Guards the PYTHONHASHSEED trap: builtin hash() on a str is salted per
    process, so a seed derived from it would differ in every spawned worker and
    break reproducibility silently. Compute the same seeds under two different
    hash salts and require identical results.
    """
    snippet = (
        "import sys; sys.path.insert(0, '.');"
        "from nas_ts.utils.seeding import derive_task_seed;"
        "print([derive_task_seed(42, i) for i in (0, 7, 499)],"
        "      derive_task_seed(42, 'indiv-7'))"
    )
    outs = []
    for salt in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt, PYTHONPATH=os.getcwd())
        outs.append(subprocess.run([sys.executable, "-c", snippet],
                                   capture_output=True, text=True,
                                   env=env, cwd=os.getcwd()).stdout.strip())
    assert outs[0] and outs[0] == outs[1], f"seed changed with PYTHONHASHSEED: {outs}"


@pytest.mark.parametrize("seed", [0, 1, 2 ** 31, 2 ** 40])
def test_seeds_stay_in_numpy_range(seed):
    """numpy rejects seeds outside [0, 2**32-1]; large configs must not crash."""
    assert 0 <= seed_everything(seed) < 2 ** 31
    assert 0 <= derive_task_seed(seed, 10) < 2 ** 31
