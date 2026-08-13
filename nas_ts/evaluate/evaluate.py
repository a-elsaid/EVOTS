# !/usr/bin/env python3
from typing import Dict, Tuple, Optional, Union
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from ..core.config import ExperimentConfig, EvalConfig
from ..core.genome_v2 import Genome
from ..models.model_builder_v2 import build_model_from_meta
from ..utils.weight_pool import WeightPool
from ..utils.devices import auto_detect_device

def _unpack_batch(batch):
    # batch can be (x,y) or (x,y,x_mark,y_mark)
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            return x, y, None, None
        if len(batch) == 4:
            x, y, x_mark, y_mark = batch
            return x, y, x_mark, y_mark
    raise ValueError(f"Unexpected batch format: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")


def _align_pred_target(y_hat: torch.Tensor, y: torch.Tensor):
    """
    Align prediction and target shapes along time and feature dims.
    Returns (y_hat_aligned, y_aligned).

    - Assumes batch dimension matches.
    - Uses the minimum length along time and feature dims.
    """
    if y_hat.ndim != 3 or y.ndim != 3:
        # fall back to naive behavior for non-3D tensors
        return y_hat, y

    B1, Lp, Dp = y_hat.shape
    B2, Lt, Dt = y.shape
    if B1 != B2:
        raise ValueError(f"Batch size mismatch: pred {B1}, target {B2}")

    L_common = min(Lp, Lt)
    D_common = min(Dp, Dt)

    if L_common == 0 or D_common == 0:
        raise ValueError(f"Common length is zero: pred={y_hat.shape}, target={y.shape}")

    y_hat_aligned = y_hat[:, :L_common, :D_common]
    y_aligned = y[:, :L_common, :D_common]
    return y_hat_aligned, y_aligned


def evaluate_genome(
    genome: Genome,
    exp_cfg: ExperimentConfig,
    weight_pool: Optional[WeightPool] = None,
    return_state: bool = False,
    init_state_dict_cpu: Optional[dict] = None,
) -> Union[Dict[str, float], tuple[Dict[str, float], dict, dict]]:
    """
    Main entry point for workers / GA.
    For now, evaluate on the first dataset only, with optional weight reuse.
    """
    out = _train_and_eval_on_dataset(
        genome, exp_cfg,
        ds_index=0,
        weight_pool=weight_pool,
        return_state=return_state,
        init_state_dict_cpu=init_state_dict_cpu,
    )
    return out

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mse_loss(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat, y = _align_pred_target(y_hat, y)
    return torch.mean((y_hat - y) ** 2).item()


def mae_loss(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    y_hat, y = _align_pred_target(y_hat, y)
    return torch.mean(torch.abs(y_hat - y)).item()


def _get_optimizer(model: nn.Module, eval_cfg: EvalConfig):
    opt_name = eval_cfg.optimizer.lower()
    kwargs = eval_cfg.optimizer_kwargs.copy()

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {eval_cfg.optimizer}")


def _load_state_checked(model: nn.Module, state: dict, *, where: str) -> None:
    """load_state_dict(strict=False) that refuses to drop keys silently.

    Every load in this file used to swallow mismatches, which is how a tokenizer
    projection could be discarded on load and re-initialised at random without a
    single line of output.
    """
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning(
            f"[Load:{where}] state_dict mismatch — "
            f"missing={list(incompatible.missing_keys)} "
            f"unexpected={list(incompatible.unexpected_keys)}"
        )



def _train_and_eval_on_dataset(
    genome: Genome,
    exp_cfg: ExperimentConfig,
    ds_index: int = 0,
    weight_pool: Optional[WeightPool] = None,
    return_state: bool = False,
    init_state_dict_cpu: Optional[dict] = None,
) -> Union[Dict[str, float], tuple[Dict[str, float], dict, dict]]:

    eval_cfg = exp_cfg.eval_config
    task = eval_cfg.task
    device = auto_detect_device(eval_cfg.device)

    ds_cfg = eval_cfg.datasets[ds_index]

    can_early_stop = eval_cfg.early_stopping
    early_stop_patience = eval_cfg.early_stop_patience
    early_stop_min_delta = eval_cfg.early_stop_min_delta
    val_check_every = eval_cfg.early_stop_checks
    

    train_loader, val_loader, test_loader, meta = ds_cfg.loader_fn(**ds_cfg.loader_kwargs)

    # Build model
    model = build_model_from_meta(genome, exp_cfg.eval_config.task, meta)
    model.to(device)

    if init_state_dict_cpu is not None:
        try:
            _load_state_checked(model, init_state_dict_cpu, where="WeightInherit")
            logger.info("[WeightInherit] Loaded init state_dict into model")
        except Exception as e:
            logger.warning(f"[WeightInherit] Failed to load init weights: {e}")

    # --- weight reuse from pool ---
    # if weight_pool is not None:
    #     state = weight_pool.get(genome)
    #     if state is not None:
    #         logger.info("[WeightPool] Reusing weights for architecture")
    #         try:
    #             model.load_state_dict(state, strict=False)
    #         except RuntimeError:
    #             pass
    #     else:
    #         logger.info("[WeightPool] No weights found for architecture; training from scratch")

    optimizer = _get_optimizer(model, eval_cfg)

    # -------------------------
    # Helpers
    # -------------------------
    def _val_mse_only() -> float:
        """Compute mean val MSE (aligned) for early stopping."""
        model.eval()
        mse_vals = []
        with torch.no_grad():
            for batch in val_loader:
                x, y, x_mark, y_mark = _unpack_batch(batch)
                x = x.to(device)
                y = y.to(device)
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)
                y_hat = model(x, x_mark=x_mark, y_mark=y_mark)  # even if y_mark unused
                y_hat_aligned, y_aligned = _align_pred_target(y_hat, y)
                mse_vals.append(torch.mean((y_hat_aligned - y_aligned) ** 2).item())
        model.train()
        if not mse_vals:
            return float("inf")
        return float(sum(mse_vals) / len(mse_vals))

    # -------------------------
    # Training + Early Stopping
    # -------------------------
    epochs = int(eval_cfg.training_steps)

    best_val = float("inf")
    best_state = None
    bad_checks = 0

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for batch in train_loader:
            x, y, x_mark, y_mark = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            optimizer.zero_grad()
            y_hat = model(x, x_mark=x_mark, y_mark=y_mark)  # even if y_mark unused
            y_hat, y = _align_pred_target(y_hat, y)
            loss = torch.mean((y_hat - y) ** 2)
            loss.backward()
            optimizer.step()

        # ---- validation (once per epoch) ----
        if can_early_stop and ((epoch + 1) % val_check_every == 0):
            cur_val = _val_mse_only()

            if cur_val < best_val - early_stop_min_delta:
                best_val = cur_val
                bad_checks = 0
                best_state = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            else:
                bad_checks += 1

            if bad_checks >= early_stop_patience:
                logger.info(
                    f"[EarlyStop] epoch={epoch+1}, "
                    f"best_val_mse={best_val:.6f}"
                )
                break
        
    elapsed = time.time() - start_time
    logger.debug(f"Training completed in {elapsed:.2f} seconds over {epoch+1} epochs.")

    # restore best weights if we have them
    if best_state is not None:
        _load_state_checked(model, best_state, where="EarlyStopRestore")

    # -------------------------
    # Final Validation Metrics (+ latency)
    # -------------------------
    model.eval()
    mse_vals = []
    mae_vals = []

    latency = None
    measured_latency = False

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x, y, x_mark, y_mark = _unpack_batch(batch)
            x = x.to(device)
            y = y.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            if not measured_latency:
                _ = model(x, x_mark=x_mark, y_mark=y_mark)  # even if y_mark unused
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.time()
                _ = model(x, x_mark=x_mark, y_mark=y_mark)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t1 = time.time()
                latency = (t1 - t0) / max(1, x.size(0))
                measured_latency = True

            y_hat = model(x, x_mark=x_mark, y_mark=y_mark)
            mse_vals.append(mse_loss(y_hat, y))
            mae_vals.append(mae_loss(y_hat, y))

    mse_mean = float(sum(mse_vals) / len(mse_vals)) if mse_vals else float("inf")
    mae_mean = float(sum(mae_vals) / len(mae_vals)) if mae_vals else float("inf")
    params = count_parameters(model)

    metrics = {
        "mse": mse_mean,
        "mae": mae_mean,
        "params": float(params),
    }
    if latency is not None:
        metrics["latency"] = float(latency)

    # (optional) also expose best_val seen during training
    if best_val != float("inf"):
        metrics["best_val_mse"] = float(best_val)

    # --- update weight pool with BEST weights ---
    # if weight_pool is not None:
    #     weight_pool.update(genome, model.state_dict())

    if return_state:
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return metrics, state_dict_cpu, meta

    return metrics


def evaluate_genome_multi(
    genome: Genome,
    exp_cfg: ExperimentConfig,
    return_state: bool = False,
):
    all_metrics = []
    first_state = None
    first_meta = None

    for ds_index in range(len(exp_cfg.eval_config.datasets)):
        out = _train_and_eval_on_dataset(genome, exp_cfg, ds_index, return_state=return_state)
        if return_state:
            m, st, meta = out
            if ds_index == 0:
                first_state, first_meta = st, meta
        else:
            m = out
        all_metrics.append(m)

    keys = all_metrics[0].keys()
    agg_metrics = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg_metrics[k] = float(sum(vals) / len(vals)) if vals else float("inf")

    if return_state:
        return agg_metrics, first_state, first_meta
    return agg_metrics



def finetune_and_test(
    best_fitness: float,
    genome: Genome,
    exp_cfg: ExperimentConfig,
    initial_state_dict_cpu: dict,
    extra_training_steps: int,
) -> tuple[dict, dict]:
    """
    Continue training from state_dict on TRAIN loader with early stopping,
    then evaluate BEST model on TEST loader.
    """
    eval_cfg = exp_cfg.eval_config
    task = eval_cfg.task
    device = auto_detect_device(eval_cfg.device)


    can_early_stop = eval_cfg.early_stopping
    early_stop_patience = eval_cfg.early_stop_patience * 2
    early_stop_min_delta = eval_cfg.early_stop_min_delta
    val_check_every = eval_cfg.early_stop_checks

    logger.info(
        f"[Finetune] Starting finetune on device={device}, "
        f"max_steps={extra_training_steps}, "
        f"patience={early_stop_patience}"
    )

    ds_cfg = eval_cfg.datasets[0]
    train_loader, val_loader, test_loader, meta = ds_cfg.loader_fn(**ds_cfg.loader_kwargs)
    

    model = build_model_from_meta(genome, task, meta)
    model.to(device)
    if initial_state_dict_cpu is not None:
        # Rebuilt from the genome, so any mismatch is a real architecture/weights
        # disagreement and must not be papered over.
        model.load_state_dict(initial_state_dict_cpu, strict=True)

    optimizer = _get_optimizer(model, eval_cfg)
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()

    # best_val_mse = float("inf")
    best_val_mse = best_fitness if best_fitness is not None else float("inf")
    search_best_val_mse = best_val_mse
    best_state_cpu = None
    patience_left = early_stop_patience

    model.train()
    train_iter = iter(train_loader)

    for step in range(1, extra_training_steps + 1):
        # ---- training step ----
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        x, y, x_mark, y_mark = _unpack_batch(batch)
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        optimizer.zero_grad()

        y_hat = model(x, x_mark=x_mark, y_mark=y_mark   )  # even if y_mark unused
        y_hat, y = _align_pred_target(y_hat, y)

        loss = mse_crit(y_hat, y)
        loss.backward()
        optimizer.step()

        # ---- validation ----
        if step % val_check_every != 0:
            continue

        model.eval()
        val_mse_vals = []

        with torch.no_grad():
            for batch in val_loader:
                vx, vy, vx_mark, vy_mark = _unpack_batch(batch)
                vx, vy, vx_mark, vy_mark = vx.to(device), vy.to(device), vx_mark.to(device), vy_mark.to(device)
                vy_hat = model(vx, x_mark=vx_mark, y_mark=vy_mark)
                vy_hat, vy = _align_pred_target(vy_hat, vy)
                val_mse_vals.append(mse_crit(vy_hat, vy).item())

        model.train()

        val_mse = float(sum(val_mse_vals) / len(val_mse_vals)) if val_mse_vals else float("inf")

        # ---- early stopping logic ----
        if val_mse < best_val_mse - early_stop_min_delta:
            search_best_val_mse = val_mse # to avoid logging confusion in the next elif
            best_val_mse = val_mse
            best_state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = early_stop_patience

            logger.info(
                f"[Finetune] Step {step:5d} | "
                f"val_mse improved → {best_val_mse:.6f}"
            )
        elif val_mse > search_best_val_mse :
            logger.debug(
                f"[Finetune] Step {step:5d}/{extra_training_steps} | "
                f"val_mse improved (search) → {val_mse:.6f}"
                f" (Search Best: {search_best_val_mse:.6f})"
            )
        else:
            patience_left -= 1
            logger.debug(
                f"[Finetune] Step {step:5d}/{extra_training_steps} | "
                f"val_mse={val_mse:.6f} | "
                f"patience left={patience_left}"
            )

            if patience_left <= 0:
                logger.info(
                    f"[Finetune] Early stopping triggered at step {step} "
                    f"(best_val_mse={best_val_mse:.6f})"
                )
                break

    # ---- restore best weights ----
    if best_state_cpu is not None:
        _load_state_checked(model, best_state_cpu, where="FinetuneRestore")
        logger.info(
            f"[Finetune] Restored best model "
            f"(val_mse={best_val_mse:.6f})"
        )
    else:
        best_state_cpu = initial_state_dict_cpu
        logger.warning("[Finetune] No validation improvement recorded; using last model")

    # ---- test evaluation ----
    model.eval()
    mse_vals, mae_vals = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y, x_mark, y_mark = _unpack_batch(batch)
            x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
            y_hat = model(x, x_mark=x_mark, y_mark=y_mark)
            y_hat, y = _align_pred_target(y_hat, y)
            mse_vals.append(mse_crit(y_hat, y).item())
            mae_vals.append(mae_crit(y_hat, y).item())

    test_mse = float(sum(mse_vals) / len(mse_vals)) if mse_vals else float("inf")
    test_mae = float(sum(mae_vals) / len(mae_vals)) if mae_vals else float("inf")

    logger.info(
        f"[Finetune] TEST results | "
        f"mse={test_mse:.6f} | mae={test_mae:.6f}"
    )

    test_metrics = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "best_val_mse": best_val_mse,
        "finetune_steps_run": step,
    }

    return test_metrics, best_state_cpu, meta
