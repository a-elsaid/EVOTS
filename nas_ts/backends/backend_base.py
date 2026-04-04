# !/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

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