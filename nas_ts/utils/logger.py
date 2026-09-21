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


# The metric the search is actually optimising, per task type: aggregate_mse
# uses metrics["mse"] for forecasting and aggregate_classification_fitness uses
# metrics["loss"] for classification (see core/config_loader.py). Logging the
# same key keeps the logged "best" identical to the fitness the engine selected
# on. The two never co-occur in one metrics dict, so probing in order is
# unambiguous.
_PRIMARY_METRIC_KEYS = ("mse", "loss")


def _primary_metric(metrics: dict):
    """Return (key, value) for the optimised metric, or (None, None) if absent."""
    for key in _PRIMARY_METRIC_KEYS:
        value = (metrics or {}).get(key)
        if value is not None:
            return key, value
    return None, None


def _ensure_csv_header(path: Path, header: list) -> None:
    """
    Create the CSV with this header, or warn if an existing file has a different
    one. Appending rows under a stale header would misalign every column, and a
    misaligned CSV is indistinguishable from a correct one downstream.
    """
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)
        return

    with open(path, "r", newline="") as f:
        existing = next(csv.reader(f), [])
    if existing and existing != header:
        logger.warning(
            f"[Logger] {path} has header {existing} but this run writes {header}. "
            f"Rows appended now will not line up with the existing ones; move or "
            f"delete the old file to get a clean CSV."
        )


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

        # metric_name says which metric metric_value holds, so one schema serves
        # both task types; accuracy and mae are the per-task extras and stay
        # blank for the task that does not produce them.
        self.eval_csv_header = [
            "eval_id",
            "generation",
            "ind_id",
            "fitness",
            "metric_name",
            "metric_value",
            "best_val_metric",
            "accuracy",
            "mae",
            "params",
            "timestamp",
        ]
        self.gen_csv_header = [
            "generation",
            "best_fitness",
            "avg_fitness",
            "metric_name",
            "best_metric",
            "avg_metric",
            "pop_size",
            "timestamp",
        ]

        _ensure_csv_header(self.eval_csv_path, self.eval_csv_header)
        _ensure_csv_header(self.gen_csv_path, self.gen_csv_header)

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
        timestamp = datetime.now(timezone.utc).isoformat()

        metric_name, metric_value = _primary_metric(safe_metrics)
        if metric_name is None:
            # Still record the row: the evaluation happened, and dropping it is
            # how the evaluations CSV ends up empty with nothing to explain it.
            logger.warning(
                f"[Logger] Individual {ind.id} has none of {_PRIMARY_METRIC_KEYS} in its "
                f"metrics (keys={sorted(safe_metrics)}). Logging the evaluation with an "
                f"empty metric value."
            )
            best_val_metric = None
        else:
            best_val_metric = safe_metrics.get(f"best_val_{metric_name}")

        with open(self.eval_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                safe_metrics.get("eval_id", -1),
                safe_metrics.get("generation", None),
                ind.id,
                ind.fitness,
                metric_name,
                metric_value,
                best_val_metric,
                safe_metrics.get("accuracy", None),
                safe_metrics.get("mae", None),
                safe_metrics.get("params", None),
                timestamp,
            ])

        with open(self.json_path, "a") as f:
            f.write(json.dumps({
                "type": "evaluation",
                "eval_id": safe_metrics.get("eval_id", -1),
                "generation": safe_metrics.get("generation", None),
                "ind_id": ind.id,
                "fitness": ind.fitness,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "best_val_metric": best_val_metric,
                "metrics": safe_metrics,
                "timestamp": timestamp,
            }) + "\n")

    def log_generation(self, population, gen_number: int):
        """Aggregate pop stats each generation."""
        inds = [ind for ind in population.get_all() if ind.fitness is not None]

        if inds:
            best_fitness = min(ind.fitness for ind in inds)
            avg_fitness = sum(ind.fitness for ind in inds) / len(inds)
        else:
            best_fitness, avg_fitness = None, None

        # The whole population shares a task, so the first individual carrying a
        # primary metric fixes the key the rest are aggregated on.
        metric_name = None
        for ind in inds:
            metric_name, _ = _primary_metric(getattr(ind, "metrics", None) or {})
            if metric_name is not None:
                break

        values, skipped = [], []
        if metric_name is not None:
            for ind in inds:
                value = (getattr(ind, "metrics", None) or {}).get(metric_name)
                if value is None:
                    skipped.append(ind.id)
                else:
                    values.append(value)

        if skipped:
            logger.warning(
                f"[Logger] Generation {gen_number}: {len(skipped)} individual(s) missing "
                f"'{metric_name}' were left out of best/avg (ids={skipped})."
            )

        if values:
            best_metric = min(values)
            avg_metric = sum(values) / len(values)
        else:
            if inds:
                logger.warning(
                    f"[Logger] Generation {gen_number}: no individual reported any of "
                    f"{_PRIMARY_METRIC_KEYS}; best/avg metric left empty."
                )
            best_metric, avg_metric = None, None

        timestamp = datetime.now(timezone.utc).isoformat()

        with open(self.gen_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                gen_number,
                best_fitness,
                avg_fitness,
                metric_name,
                best_metric,
                avg_metric,
                population.size(),
                timestamp,
            ])

        with open(self.json_path, "a") as f:
            f.write(json.dumps({
                "type": "generation",
                "generation": gen_number,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "metric_name": metric_name,
                "best_metric": best_metric,
                "avg_metric": avg_metric,
                "pop_size": population.size(),
                "timestamp": timestamp,
            }) + "\n")
