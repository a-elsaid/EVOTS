import csv
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger
import yaml
from typing import Union
import sys

try:
    import torch
except Exception:
    torch = None


def setup_logging(logs_dir: str):
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # stdout works better than stderr on cluster with Rich Progress
    logger.add(
        sys.stdout,
        level="DEBUG",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    # File logging
    logger.add(
        log_dir / "runtime.log",
        level="DEBUG",
        rotation="100 MB",
        enqueue=True,      # also safe for threads
    )


def _json_safe_metrics(metrics: dict) -> dict:
    """
    Keep only JSON-safe scalar-ish metrics.
    Drops private keys like '_state_dict_cpu', '_meta', etc.
    Converts scalar tensors to Python floats if they sneak in.
    """
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(k, str) and k.startswith("_"):
            continue  # drop big/non-serializable payloads

        # Convert scalar tensors to float
        try:
            import torch
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = float(v.item())
                else:
                    continue
        except Exception:
            pass

        # keep simple JSON types
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v

    return out

class Logger:
    """
    Writes:
      - <run_name>_evaluations.csv
      - <run_name>_generations.csv
      - <run_name>.jsonl  (JSONL = one JSON object per line)
    Also can copy the YAML config used for the run into the same directory.
    """

    def __init__(self, log_dir: str, run_name: str = "experiment"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = run_name  # IMPORTANT (used by save_config_copy)

        self.eval_csv_path = self.log_dir / f"{run_name}_evaluations.csv"
        self.gen_csv_path  = self.log_dir / f"{run_name}_generations.csv"
        self.json_path     = self.log_dir / f"{run_name}.jsonl"

        if not self.eval_csv_path.exists():
            with open(self.eval_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "eval_id",
                    "generation",
                    "ind_id",
                    "fitness",
                    "result_mse",
                    "best_mse",
                    "loss",
                    "mae",
                    "params",
                    "timestamp",
                ])

        if not self.gen_csv_path.exists():
            with open(self.gen_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "generation",
                    "best_fitness",
                    "avg_fitness",
                    "best_mse",
                    "avg_mse",
                    "pop_size",
                    "timestamp",
                ])

    def save_resolved_config(self, cfg: dict) -> Path:
        """
        Save the FINAL configuration actually used for this run
        (after --set overrides are applied).
        """
        path = self.log_dir / f"{self.run_name}__resolved.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return path

    def save_config_copy(self, yaml_path: Union[str, Path]) -> Path:
        """
        Copy YAML file into the SAME directory as CSV/JSON results.
        Output example: logs/etth1_test__config.yaml
        """
        yaml_path = Path(yaml_path)
        dst = self.log_dir / f"{self.run_name}__config{yaml_path.suffix or '.yaml'}"
        shutil.copyfile(yaml_path, dst)
        return dst

    def log_evaluation(self, ind):
        """Write a single evaluation event: individual metrics."""
        metrics = ind.metrics or {}

        safe_metrics = _json_safe_metrics(metrics)

        with open(self.json_path, "a") as f:
            f.write(json.dumps({
                "type": "evaluation",
                "eval_id": safe_metrics.get("eval_id", -1),
                "generation": safe_metrics.get("generation", None),
                "ind_id": ind.id,
                "fitness": ind.fitness,
                "result_mse": safe_metrics.get("result_mse", None),
                "best_mse": safe_metrics.get("best_mse", None),
                "metrics": safe_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }) + "\n")

    def log_generation(self, population, gen_number: int):
        """Aggregate pop stats each generation."""
        inds = [ind for ind in population.get_all() if ind.fitness is not None]

        if inds:
            best_fitness = min(ind.fitness for ind in inds)
            avg_fitness = sum(ind.fitness for ind in inds) / len(inds)
        else:
            best_fitness, avg_fitness = None, None

        mse_inds = [ind for ind in inds if getattr(ind, "metrics", None) and ind.metrics.get("mse") is not None]
        if mse_inds:
            best_mse = min(ind.metrics["mse"] for ind in mse_inds)
            avg_mse = sum(ind.metrics["mse"] for ind in mse_inds) / len(mse_inds)
        else:
            best_mse, avg_mse = None, None

        with open(self.gen_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                gen_number,
                best_fitness,
                avg_fitness,
                best_mse,
                avg_mse,
                population.size(),
                datetime.now(timezone.utc).isoformat()
            ])

        with open(self.json_path, "a") as f:
            f.write(json.dumps({
                "type": "generation",
                "generation": gen_number,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_mse": best_mse,
                "avg_mse": avg_mse,
                "pop_size": population.size(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }) + "\n")
