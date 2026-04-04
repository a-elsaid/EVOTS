import random
from typing import List

from ..core.individual import Individual
from ..core.config import SelectionConfig
from ..core.population import Population

def tournament_selection(pop, sel_cfg: SelectionConfig, k: int) -> Individual:
    """
    Select the best individual from a random sample of size k.
    Robust to small populations: clamps k <= |pop|.
    """
    individuals = pop.get_all()
    n = len(individuals)
    if n == 0:
        raise RuntimeError("Cannot do selection from an empty population")
    if n == 1:
        return individuals[0]

    k = max(1, min(k, n))  # clamp k to [1, n]

    subset = random.sample(individuals, k)

    if sel_cfg.mode == "single":
        best = min(subset, key=lambda ind: ind.fitness)
        return best
    else:
        # multi-objective: use Pareto + crowding
        front = pareto_front(subset, sel_cfg.objectives)
        if len(front) == 1:
            return front[0]
        cd = crowding_distances(front, sel_cfg.objectives)
        return max(cd.items(), key=lambda x: x[1])[0]
    

def dominates(a: Individual, b: Individual, objectives: List[str]) -> bool:
    """
    a dominates b if a is <= b on all objectives, and < on at least one.
    """
    a_metrics = a.metrics
    b_metrics = b.metrics

    better_or_equal = all(a_metrics[obj] <= b_metrics[obj] for obj in objectives)
    strictly_better = any(a_metrics[obj] < b_metrics[obj] for obj in objectives)
    return better_or_equal and strictly_better


def pareto_front(indivs: List[Individual], objectives: List[str]) -> List[Individual]:
    front = []
    for a in indivs:
        if not any(dominates(b, a, objectives) for b in indivs if b is not a):
            front.append(a)
    return front


def crowding_distances(front: List[Individual], objectives: List[str]):
    """
    NSGA-II style crowding distance.
    Returns dict: Individual -> distance
    """
    if not front:
        return {}

    distances = {ind: 0.0 for ind in front}
    N = len(front)

    for obj in objectives:
        front_sorted = sorted(front, key=lambda ind: ind.metrics[obj])
        distances[front_sorted[0]] = float("inf")
        distances[front_sorted[-1]] = float("inf")

        obj_values = [ind.metrics[obj] for ind in front_sorted]
        minv, maxv = obj_values[0], obj_values[-1]
        if maxv == minv:
            continue

        for i in range(1, N - 1):
            prev = obj_values[i - 1]
            next = obj_values[i + 1]
            distances[front_sorted[i]] += (next - prev) / (maxv - minv)

    return distances


def find_worst_single(pop: Population) -> Individual:
    return max(pop.get_all(), key=lambda ind: ind.fitness)


def find_worst_multi(pop: Population, sel_cfg: SelectionConfig) -> Individual:
    individuals = pop.get_all()

    front = pareto_front(individuals, sel_cfg.objectives)

    dominated = [ind for ind in individuals if ind not in front]

    if dominated:
        cd = crowding_distances(dominated, sel_cfg.objectives)
        return min(cd.items(), key=lambda x: x[1])[0]

    else:
        # If everyone is nondominated (rare, small pop),
        # remove one with lowest overall crowding distance in the front.
        cd = crowding_distances(front, sel_cfg.objectives)
        return min(cd.items(), key=lambda x: x[1])[0]
    

def find_worst(pop: Population, sel_cfg: SelectionConfig) -> Individual:
    if sel_cfg.mode == "single":
        return find_worst_single(pop)
    else:
        return find_worst_multi(pop, sel_cfg)
