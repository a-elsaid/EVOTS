# !/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


def failure_metrics(exp_cfg: Optional[Any], error: BaseException) -> Dict[str, Any]:
    """
    Metrics for a genome whose evaluation raised, keyed for the task in use.

    The fitness aggregator reads the task's own metric ("loss" for
    classification, "mse" for forecasting), so a failure dict carrying the wrong
    key raises KeyError in the engine and one crashed genome takes down the whole
    search. inf is the correct score for a failure either way: every comparison
    in the engine treats lower as better.
    """
    task_type = None
    if exp_cfg is not None:
        task_type = getattr(getattr(getattr(exp_cfg, "eval_config", None), "task", None),
                            "task_type", None)

    if task_type == "classification":
        metrics: Dict[str, Any] = {"loss": float("inf"), "params": float("inf")}
    else:
        metrics = {"mse": float("inf"), "mae": float("inf"), "params": float("inf")}

    metrics["worker_error"] = str(error)
    return metrics


class EvaluationBackend(ABC):
    """
    Abstract interface.

    You can implement this with threads, processes, MPI, Ray, etc.
    """

    @abstractmethod
    def submit(self, indiv_id: str, genome: Any):
        """
        Submit a genome to be evaluated asynchronously.
        """
        pass

    @abstractmethod
    def poll(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Non-blocking: return a list of completed evaluations.

        Each element: (indiv_id, metrics_dict)
        """
        pass

    def shutdown(self, wait: bool = True) -> None:
        """Release any worker processes / threads. Safe to call multiple times."""
        pass