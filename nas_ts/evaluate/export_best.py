import torch
from loguru import logger

from ..models.model_builder_v2 import build_model_from_meta
from ..core.genome_v2 import Genome


def export_best_portable(best_pt_path: str, out_ts_path: str, exp_cfg) -> bool:
    """
    Loads the best checkpoint saved by engine._save_best() (weights-only format),
    rebuilds the model, then exports TorchScript (script first, trace fallback).

    This checkpoint format avoids pickling Genome objects.
    """
    ckpt = torch.load(best_pt_path, map_location="cpu", weights_only=True)
    genome = Genome.from_dict(ckpt["genome_dict"])
    state = ckpt["state_dict"]
    meta = ckpt["meta"]

    model = build_model_from_meta(genome, exp_cfg.eval_config.task, meta)
    model.load_state_dict(state, strict=False)
    model.eval()


    # Try scripting first (best)
    try:
        ts = torch.jit.script(model)
        ts.save(out_ts_path)
        logger.info(f"[Export] TorchScript scripted OK: {out_ts_path}")
        return True
    except Exception as e:
        logger.warning(f"[Export] script() failed, will try trace: {e}")

    # Trace fallback (best-effort; disable check_trace for your models)
    B = 1
    L = exp_cfg.eval_config.task.input_length
    C = meta.get("d_in", exp_cfg.eval_config.task.d_in)
    example = torch.randn(B, L, C)

    ts = torch.jit.trace(model, example, strict=False, check_trace=False)
    ts.save(out_ts_path)
    logger.info(f"[Export] TorchScript traced OK (no checks): {out_ts_path}")
    return True