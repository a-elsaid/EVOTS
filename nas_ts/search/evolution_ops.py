import random
import copy
from typing import Optional

from ..core.config import SearchSpaceConfig, GenomeConstraints, EvolutionConfig
from ..core.genome_v2 import Genome
from .mutation import mutate_genome
from .crossover import crossover_genome




# -------------------------
# reproduce() = crossover + mutation
# -------------------------

def reproduce(
    parent1: Genome,
    parent2: Genome,
    ss: SearchSpaceConfig,
    evo: EvolutionConfig,
    constraints: Optional[GenomeConstraints] = None
) -> Genome:
    """
    Crossover + mutation wrapper used by the evolutionary engine.
    """

    if random.random() < evo.crossover_rate:
        child = crossover_genome(parent1, parent2, ss, evo, constraints)
    else:
        child = copy.deepcopy(random.choice([parent1, parent2]))

    child = mutate_genome(child, ss, evo, constraints)

    return child