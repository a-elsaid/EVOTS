"""
Process-wide RNG seeding.

Nothing seeded random, numpy or torch anywhere in the codebase: only the data
split was seeded, through data.tabular.random_seed. evo.random_seed existed in
every config and was read by nobody, so two runs of the same config explored
different architectures and trained from different weights. That makes any
multi-seed experiment uncontrolled, and it applies to forecasting exactly as much
as to classification.

Deliberately does NOT enable torch.use_deterministic_algorithms: it raises on
several CUDA kernels, which would turn a reproducibility aid into a crash on the
cluster.

What reproducibility this buys, precisely:

  * Per-genome results reproduce at any worker count. A genome's evaluation is
    seeded from its own ID, so it trains identically whichever worker picks it up.
  * A whole run reproduces only at num_workers=1. The search is asynchronous and
    steady-state: completion order decides which individuals are in the
    population when a child is bred, so with several workers the parents
    available at each spawn differ between runs, and the architectures explored
    diverge -- not only after the initial population but throughout. Seeding
    cannot fix that; it is a property of the search, not of the RNG.
"""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch
from loguru import logger

# Keeps derived seeds inside the range numpy accepts for a seed.
_SEED_MODULUS = 2 ** 31 - 1

# Room for this many individuals per run before task seeds wrap into the next
# base seed's block. 500 genomes is the EXAQC-matched budget, so this is ample.
_TASK_TOKEN_SPACE = 1_000_000


def seed_everything(seed: int, *, where: str = "Main") -> int:
    """Seed random, numpy and torch (CPU and CUDA) in this process."""
    seed = int(seed) % _SEED_MODULUS

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"[Seed] {where} seeded with {seed}")
    return seed


def derive_task_seed(base_seed: int, indiv_id) -> int:
    """
    The seed for evaluating one individual: a function of the base seed and the
    individual's ID, and of nothing else.

    Deliberately independent of which worker runs the task. Seeding per worker
    instead makes a genome's trained weights depend on pool scheduling, so the
    same genome evaluated by a different worker gives different numbers.

    Never uses the builtin hash() on the ID: it is salted per process via
    PYTHONHASHSEED, so every spawned worker would compute a different value for
    the same ID and reproducibility would break silently. Integer IDs (what
    Individual.id actually is) are used directly; anything else goes through
    crc32, which is stable across processes and runs.

    Spacing is collision-free: distinct (base_seed, indiv_id) pairs give distinct
    seeds while base_seed * _TASK_TOKEN_SPACE stays inside the modulus, which
    covers every base seed up to ~2146 with up to a million individuals each.
    """
    if isinstance(indiv_id, (int, np.integer)) and not isinstance(indiv_id, bool):
        token = int(indiv_id) % _TASK_TOKEN_SPACE
    else:
        token = zlib.crc32(str(indiv_id).encode("utf-8")) % _TASK_TOKEN_SPACE

    return (int(base_seed) * _TASK_TOKEN_SPACE + token + 1) % _SEED_MODULUS
