from __future__ import annotations

import os
import sys
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, Future

import torch
from loguru import logger

from .backend_base import EvaluationBackend
from ..core.config import ExperimentConfig
from ..evaluate.evaluate import evaluate_genome
from ..utils.devices import validate_gpu_ids
from ..utils.logger import setup_logging


_G_EXP_CFG: Optional[ExperimentConfig] = None
_G_GPU_ID: Optional[int] = None


def auto_detect_gpu_ids() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _init_process(exp_cfg: ExperimentConfig, gpu_id: Optional[int]):
    global _G_EXP_CFG, _G_GPU_ID
    _G_EXP_CFG = exp_cfg
    _G_GPU_ID = gpu_id

    setup_logging(exp_cfg.run_info.logs_dir + "/" + exp_cfg.run_info.name)
    torch.set_num_threads(1)

    if gpu_id is not None:
        if not torch.cuda.is_available():
            logger.warning(f"[Worker] Requested gpu_id={gpu_id} but CUDA not available. Using CPU.")
            _G_GPU_ID = None
            return

        n = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n:
            raise ValueError(f"[Worker] gpu_id={gpu_id} out of range. Visible CUDA devices={n}")

        torch.cuda.set_device(gpu_id)
        _ = torch.empty(1, device=f"cuda:{gpu_id}")
        logger.info(f"[Worker] PID={os.getpid()} bound to cuda:{gpu_id}")
    else:
        logger.info(f"[Worker] PID={os.getpid()} using CPU")


def _process_worker(indiv_id: str, genome: Any) -> Tuple[str, Dict[str, float]]:
    try:
        out = evaluate_genome(
            genome,
            _G_EXP_CFG,
            return_state=False,
        )

        if isinstance(out, dict):
            clean = {}
            for k, v in out.items():
                if torch.is_tensor(v):
                    clean[k] = float(v.detach().cpu().item()) if v.numel() == 1 else float("nan")
                else:
                    clean[k] = v
            out = clean

        return indiv_id, out

    except Exception as e:
        logger.exception(f"[Worker] crash indiv_id={indiv_id}")
        return indiv_id, {
            "mse": float("inf"),
            "mae": float("inf"),
            "params": float("inf"),
            "worker_error": str(e),
        }


class ProcessBackend(EvaluationBackend):
    def __init__(
        self,
        exp_cfg: ExperimentConfig,
        max_workers: int = 4,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.exp_cfg = exp_cfg
        self.max_workers = int(max_workers)
        self._shutdown = False

        preferred = str(getattr(self.exp_cfg.eval_config, "device", "auto") or "auto").lower()

        if preferred != "auto":
            logger.info(f"[ProcessBackend] Bypassing GPU pinning (eval.device='{preferred}')")
            self.gpu_ids = []
        else:
            if gpu_ids is None:
                gpu_ids = auto_detect_gpu_ids()
            gpu_ids = list(gpu_ids)
            if gpu_ids:
                gpu_ids = validate_gpu_ids(gpu_ids)
            self.gpu_ids = gpu_ids

        if self.gpu_ids and self.max_workers > len(self.gpu_ids):
            logger.warning(
                f"[ProcessBackend] max_workers={self.max_workers} > len(gpu_ids)={len(self.gpu_ids)}. "
                f"Reducing to {len(self.gpu_ids)}."
            )
            self.max_workers = len(self.gpu_ids)

        self._future_to_id: Dict[Future, str] = {}
        self._rr = 0

        if not self.gpu_ids:
            logger.info(f"[ProcessBackend] CPU mode: max_workers={self.max_workers}")
            self.executors = [
                ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    initializer=_init_process,
                    initargs=(self.exp_cfg, None),
                )
            ]
        else:
            use_gpu_ids = self.gpu_ids[:self.max_workers]
            logger.info(f"[ProcessBackend] GPU mode: gpu_ids={use_gpu_ids}")
            self.executors = [
                ProcessPoolExecutor(
                    max_workers=1,
                    initializer=_init_process,
                    initargs=(self.exp_cfg, gid),
                )
                for gid in use_gpu_ids
            ]

    def submit(self, indiv_id: str, genome: Any):
        if self._shutdown:
            raise RuntimeError("ProcessBackend.submit() called after shutdown()")
        ex = self.executors[self._rr % len(self.executors)]
        self._rr += 1
        fut = ex.submit(_process_worker, indiv_id, genome)
        self._future_to_id[fut] = indiv_id

    def poll(self) -> List[Tuple[str, Dict[str, float]]]:
        completed: List[Tuple[str, Dict[str, float]]] = []
        done = [f for f in list(self._future_to_id.keys()) if f.done()]
        for fut in done:
            completed.append(fut.result())
            del self._future_to_id[fut]
        return completed

    def pending(self) -> int:
        return len(self._future_to_id)

    def shutdown(self, wait: bool = True):
        self._shutdown = True
        for ex in self.executors:
            ex.shutdown(wait=wait)