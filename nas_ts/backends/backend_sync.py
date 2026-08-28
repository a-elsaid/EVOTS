from typing import List, Tuple, Dict, Any, Optional
from .backend_base import EvaluationBackend
from ..core.config import ExperimentConfig
from ..evaluate.evaluate import evaluate_genome
from ..utils.weight_pool import WeightPool

class SyncBackend(EvaluationBackend):
    """
    Synchronous backend: evaluate immediately on submit.
    Good for debugging and deterministic behavior.
    """

    def __init__(self, exp_cfg: ExperimentConfig, weight_pool: Optional[WeightPool] = None):
        self.exp_cfg = exp_cfg
        self.weight_pool = weight_pool
        self._completed: List[Tuple[str, Dict[str, float]]] = []

    def submit(self, indiv_id: str, genome: Any):
        metrics = evaluate_genome(genome, self.exp_cfg, weight_pool=self.weight_pool)
        self._completed.append((indiv_id, metrics))

    def poll(self) -> List[Tuple[str, Dict[str, float]]]:
        completed = self._completed
        self._completed = []
        return completed

    def shutdown(self, wait: bool = True) -> None:
        pass
    