from __future__ import annotations

import threading
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
from loguru import logger

from .backend_base import EvaluationBackend
from ..core.config import ExperimentConfig
from ..evaluate.evaluate import evaluate_genome
from ..utils.weight_pool import WeightPool


class ThreadedBackend(EvaluationBackend):
    """
    ThreadPool backend with optional GPU pinning.
    Worker IDs are stable per thread. In GPU mode, at most one eval
    runs per GPU at a time via a per-GPU semaphore.
    """

    def __init__(
        self,
        exp_cfg: ExperimentConfig,
        max_workers: int = 4,
        weight_pool: Optional[WeightPool] = None,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.exp_cfg = exp_cfg
        self.max_workers = int(max_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.weight_pool = weight_pool
        self._shutdown = False

        self._thread_to_worker_id: Dict[int, int] = {}
        self._worker_lock = threading.Lock()

        if gpu_ids is None:
            n = torch.cuda.device_count()
            self.gpu_ids: List[int] = list(range(n)) if n > 0 else []
        else:
            self.gpu_ids = list(gpu_ids)

        self._gpu_slots: Dict[int, threading.Semaphore] = {
            gid: threading.Semaphore(1) for gid in self.gpu_ids
        }

        self._future_to_id: Dict[Future, str] = {}
        self._lock = threading.Lock()

    def _get_worker_id(self) -> int:
        """Return a stable integer ID for the current executor thread."""
        tid = threading.get_ident()
        with self._worker_lock:
            if tid not in self._thread_to_worker_id:
                self._thread_to_worker_id[tid] = len(self._thread_to_worker_id)
            return self._thread_to_worker_id[tid]

    def _pick_gpu(self, worker_id: int) -> Optional[int]:
        if not self.gpu_ids:
            return None
        return self.gpu_ids[worker_id % len(self.gpu_ids)]

    def _worker(self, indiv_id: str, genome: Any) -> Tuple[str, Dict[str, float]]:
        worker_id = self._get_worker_id()
        gpu = self._pick_gpu(worker_id)

        try:
            if gpu is None:
                logger.info(f"Worker({worker_id}) CPU starting indiv {indiv_id}")
                out = evaluate_genome(genome, self.exp_cfg, weight_pool=self.weight_pool, return_state=True)
                logger.info(f"Worker({worker_id}) CPU finished indiv {indiv_id}")
            else:
                with self._gpu_slots[gpu]:
                    torch.cuda.set_device(gpu)
                    logger.info(f"Worker({worker_id}) GPU={gpu} starting indiv {indiv_id}")
                    out = evaluate_genome(genome, self.exp_cfg, weight_pool=self.weight_pool, return_state=True)
                    logger.info(f"Worker({worker_id}) GPU={gpu} finished indiv {indiv_id}")

            if isinstance(out, tuple) and len(out) == 3:
                metrics, state, meta = out
                metrics["state_dict_cpu"] = state
                metrics["meta"] = meta
                return indiv_id, metrics

            return indiv_id, out

        except Exception as e:
            logger.exception(f"Worker({worker_id}) error in indiv {indiv_id}")
            return indiv_id, {
                "mse": float("inf"),
                "mae": float("inf"),
                "params": float("inf"),
                "worker_error": str(e),
            }

    def submit(self, indiv_id: str, genome: Any):
        if self._shutdown:
            raise RuntimeError("ThreadedBackend.submit() called after shutdown()")
        future = self.executor.submit(self._worker, indiv_id, genome)
        with self._lock:
            self._future_to_id[future] = indiv_id

    def poll(self) -> List[Tuple[str, Dict[str, float]]]:
        completed: List[Tuple[str, Dict[str, float]]] = []
        done_futures: List[Future] = []

        with self._lock:
            for fut in list(self._future_to_id.keys()):
                if fut.done():
                    done_futures.append(fut)

        for fut in done_futures:
            with self._lock:
                indiv_id = self._future_to_id.pop(fut, None)
            if indiv_id is None:
                continue
            try:
                _, metrics = fut.result()
                completed.append((indiv_id, metrics))
            except Exception as e:
                logger.exception(f"[ThreadedBackend] Future failed for indiv_id={indiv_id}")
                completed.append((indiv_id, {
                    "mse": float("inf"),
                    "mae": float("inf"),
                    "params": float("inf"),
                    "worker_error": str(e),
                }))

        return completed

    def pending(self) -> int:
        with self._lock:
            return len(self._future_to_id)

    def shutdown(self, wait: bool = True):
        self._shutdown = True
        self.executor.shutdown(wait=wait)