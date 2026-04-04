# !/usr/bin/env python3


import os
import torch
from typing import Any, Optional

from ..core.population import Population
from .weight_pool import WeightPool

def save_checkpoint(
    path: str,
    population: Population,
    eval_count: int,
    weight_pool: Optional[WeightPool] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "eval_count": eval_count,
        "population": population.get_all(),  # list of Individuals
    }
    if weight_pool is not None:
        data["weight_pool"] = weight_pool._pool
    torch.save(data, path)


def load_checkpoint(path: str) -> Any:
    return torch.load(path, map_location="cpu")