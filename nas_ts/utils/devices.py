import torch
from typing import Optional

from loguru import logger

# Warn once per process: auto_detect_device is called for every evaluation and
# in every worker, and repeating this on each call would bury the run log.
_MPS_WARNED = False


def _warn_mps_once() -> None:
    global _MPS_WARNED
    if _MPS_WARNED:
        return
    _MPS_WARNED = True
    logger.warning(
        "[Device] Resolved to 'mps'. MPS is untested with the process backend: "
        "under torch 2.0.1 the spawned worker dies with BrokenProcessPool and no "
        "Python traceback. Set eval.device=cpu if the search fails to start."
    )


def validate_gpu_ids(gpu_ids: list[int]) -> list[int]:
    """
    Validate GPU IDs against *visible* CUDA devices in this process.

    - Requires CUDA to be available.
    - Checks 0 <= id < torch.cuda.device_count()
    - Deduplicates while preserving order.
    """
    if not torch.cuda.is_available():
        raise ValueError(f"gpu_ids={gpu_ids} provided but CUDA is not available")

    n = torch.cuda.device_count()
    if n <= 0:
        raise ValueError("CUDA reports 0 visible devices")

    out: list[int] = []
    seen = set()
    for gid in gpu_ids:
        if not isinstance(gid, int):
            raise TypeError(f"gpu_id must be int, got {type(gid)}: {gid}")
        if gid < 0 or gid >= n:
            raise ValueError(f"gpu_id={gid} out of range [0,{n-1}] (visible cuda devices={n})")
        if gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
    return out


def auto_detect_device(preferred: Optional[str] = "auto") -> str:
    """
    Returns a device string:
      - 'cuda:{current_device}' if CUDA available
      - else 'mps' if available
      - else 'cpu'

    If preferred is:
      - None or "auto": choose best available
      - "cuda": return current cuda device explicitly (cuda:{idx})
      - "cuda:N": return as-is
      - "cpu"/"mps": return as-is
    """
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            # respect torch.cuda.set_device(...) already done in this process
            return f"cuda:{torch.cuda.current_device()}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _warn_mps_once()
            return "mps"
        return "cpu"

    if preferred == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        return f"cuda:{torch.cuda.current_device()}"

    if preferred == "mps":
        _warn_mps_once()

    return preferred
