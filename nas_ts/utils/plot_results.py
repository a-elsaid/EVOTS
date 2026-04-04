from __future__ import annotations
from typing import Optional, Union
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import torch

from ..core.genome_v2 import Genome
from ..models.model_builder_v2 import build_model_from_meta
from ..evaluate.evaluate import _align_pred_target


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        logger.warning(f"[Plot] CSV is empty: {path}")
        return None
    return df


def _load_best_checkpoint(best_pt_path: Path):
    """
    Supports:
      - Option A: {"genome_dict": {...}, "model_state_dict": {...}, "meta": {...}}
      - Legacy:   {"genome": GenomeObj, "model_state_dict": {...}, "meta": {...}}
    """
    try:
        ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")
    except Exception:
        ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"[Plot] Missing model_state_dict. Keys={list(ckpt.keys())}")
    if "meta" not in ckpt:
        raise KeyError(f"[Plot] Missing meta. Keys={list(ckpt.keys())}")

    state = ckpt["model_state_dict"]
    meta = ckpt["meta"]

    if "genome_dict" in ckpt and isinstance(ckpt["genome_dict"], dict):
        return Genome.from_dict(ckpt["genome_dict"]), state, meta

    if "genome" in ckpt:
        g = ckpt["genome"]
        genome = Genome.from_dict(g) if isinstance(g, dict) else g
        return genome, state, meta

    raise KeyError(f"[Plot] Missing genome_dict/genome. Keys={list(ckpt.keys())}")


def plot_generations_csv(gen_csv_path: Union[str, Path], out_dir: Union[str, Path]) -> Optional[Path]:
    gen_csv_path = Path(gen_csv_path)
    out_dir = _ensure_dir(Path(out_dir))

    df = _read_csv_if_exists(gen_csv_path)
    if df is None:
        return None

    for col in ("generation", "best_fitness", "avg_fitness"):
        if col not in df.columns:
            logger.warning(f"[Plot] generations CSV missing column '{col}': {gen_csv_path}")
            return None

    fig = plt.figure()
    plt.plot(df["generation"], df["best_fitness"], label="best_fitness")
    plt.plot(df["generation"], df["avg_fitness"], label="avg_fitness")
    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    plt.title("NAS progress by generation")
    plt.legend()

    out_path = out_dir / "generations_curve.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Plot] Saved {out_path}")
    return out_path


def plot_evaluations_csv(eval_csv_path: Union[str, Path], out_dir: Union[str, Path]) -> Optional[Path]:
    eval_csv_path = Path(eval_csv_path)
    out_dir = _ensure_dir(Path(out_dir))

    df = _read_csv_if_exists(eval_csv_path)
    if df is None:
        return None

    if "eval_id" not in df.columns:
        logger.warning(f"[Plot] evaluations CSV missing 'eval_id': {eval_csv_path}")
        return None

    fig = plt.figure()
    if "fitness" in df.columns:
        plt.plot(df["eval_id"], df["fitness"], label="fitness", alpha=0.9)
    if "result_mse" in df.columns:
        plt.plot(df["eval_id"], df["result_mse"], label="result_mse", alpha=0.7)
    if "best_mse" in df.columns:
        plt.plot(df["eval_id"], df["best_mse"], label="best_mse", alpha=0.9)

    plt.xlabel("eval_id")
    plt.ylabel("value")
    plt.title("Evaluations over time")
    plt.legend()

    out_path = out_dir / "evaluations_curve.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Plot] Saved {out_path}")
    return out_path


@torch.no_grad()
def plot_predictions_from_best(
    best_pt_path: Union[str, Path],
    exp_cfg,
    out_dir: Union[str, Path],
    split: str = "test",
    num_batches: int = 1,
    max_series: int = 3,
) -> Optional[Path]:
    """
    Loads best checkpoint, rebuilds model, runs a few batches, plots pred vs true.
    Assumes dataset loader returns (train_loader, val_loader, test_loader, meta).
    """
    best_pt_path = Path(best_pt_path)
    out_dir = _ensure_dir(Path(out_dir))

    genome, state, meta = _load_best_checkpoint(best_pt_path)

    ds_cfg = exp_cfg.eval_config.datasets[0]
    loaders = ds_cfg.loader_fn(**ds_cfg.loader_kwargs)
    if len(loaders) != 4:
        raise ValueError("Expected loader_fn to return (train_loader, val_loader, test_loader, meta)")

    train_loader, val_loader, test_loader, _ = loaders
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}.get(split)
    if loader is None:
        raise ValueError(f"split must be one of train/val/test, got {split}")

    model = build_model_from_meta(genome, exp_cfg.eval_config.task, meta)
    if state is not None:
        model.load_state_dict(state, strict=False)
    else:
        logger.warning("[Plot] No model_state_dict found. Using randomly initialized model.")
    model.eval()
    model.to(torch.device("cpu"))

    fig = plt.figure()
    plotted = 0

    for b, batch in enumerate(loader):
        if b >= num_batches:
            break
        if len(batch) == 2:
            x, y = batch
            x_mark = y_mark = None
        elif len(batch) == 4:
            x, y, x_mark, y_mark = batch
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")

        try:
            y_hat = model(x, x_mark=x_mark, y_mark=y_mark)
        except TypeError:
            y_hat = model(x)

        try:
            y_hat, y = _align_pred_target(y_hat, y)
        except Exception as e:
            logger.warning(f"[Plot] _align_pred_target failed: {e}")

        for i in range(min(x.shape[0], max_series)):
            if plotted >= max_series:
                break
            plt.plot(y[i, :, 0].detach().cpu().numpy(), linestyle="--", label=f"true[{plotted}]")
            plt.plot(y_hat[i, :, 0].detach().cpu().numpy(), label=f"pred[{plotted}]")
            plotted += 1

    plt.xlabel("forecast horizon step")
    plt.ylabel("value")
    plt.title(f"Predictions vs targets ({split})")
    plt.legend(ncol=2, fontsize=8)

    out_path = out_dir / f"pred_vs_true_{split}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Plot] Saved {out_path}")
    return out_path


def plot_run(
    run_dir: Union[str, Path],
    exp_cfg=None,
    also_plot_predictions: bool = False,
    best_kind: str = "best_finetuned",  # "best" or "best_finetuned"
) -> None:
    """
    Plot all artifacts for a run.

    Expected structure:
      run_dir/
        <run_name>_evaluations.csv
        <run_name>_generations.csv
        <run_name>__best.pt
        plots/
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    run_name = run_dir.name
    plots_dir = _ensure_dir(run_dir / "plots")

    eval_csv = run_dir / f"{run_name}_evaluations.csv"
    gen_csv = run_dir / f"{run_name}_generations.csv"

    if eval_csv.exists():
        plot_evaluations_csv(eval_csv, plots_dir)
    else:
        logger.warning(f"[Plot] Missing {eval_csv}")

    if gen_csv.exists():
        plot_generations_csv(gen_csv, plots_dir)
    else:
        logger.warning(f"[Plot] Missing {gen_csv}")

    if also_plot_predictions:
        if exp_cfg is None:
            raise ValueError("exp_cfg is required to plot predictions")
        best_pt = run_dir / f"{run_name}__{best_kind}.pt"
        if best_pt.exists():
            plot_predictions_from_best(best_pt_path=best_pt, exp_cfg=exp_cfg, out_dir=plots_dir, split="test")
        else:
            logger.warning(f"[Plot] Missing {best_pt}")