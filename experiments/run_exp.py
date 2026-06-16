#!/usr/bin/env python3

import os
import torch

torch.set_num_interop_threads(1)
torch.set_num_threads(1)
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import yaml
import multiprocessing as mp
from pathlib import Path

from loguru import logger

from nas_ts.evaluate.evaluate import finetune_and_test
from nas_ts.utils.parser_args import parse_args, HELP_SET_TEXT
from nas_ts.utils.logger import setup_logging, Logger
from nas_ts.backends.backend_process import ProcessBackend as Backend
from nas_ts.search.engine import EvolutionEngine
from nas_ts.core.config_loader import load_cfg, build_experiment
from nas_ts.utils.plot_results import plot_run
from nas_ts.utils.weight_pool import WeightPool
from nas_ts.utils.devices import auto_detect_device


def print_config_summary(cfg: dict):
    logger.info("\n===== CONFIGURATION IN USE =====")
    logger.info(f"Run name: {cfg['run']['name']}")
    for section in ("evo", "eval", "task", "constraints"):
        logger.info(f"\n--- {section.upper()} ---")
        for k, v in cfg[section].items():
            logger.info(f"{section}.{k}: {v}")
    logger.info("\n--- DATA.CSV ---")
    for k, v in cfg["data"]["csv"].items():
        logger.info(f"data.csv.{k}: {v}")
    logger.info("\n--- SEARCH SPACE ---")
    for k, v in cfg["search_space"].items():
        logger.info(f"search_space.{k}: {v}")
    logger.info("\n================================\n")


def main():
    parser = parse_args()
    args = parser.parse_args()

    if args.help_set:
        print(HELP_SET_TEXT)
        sys.exit(0)

    if args.print_config and not args.config:
        parser.error("--print-config requires --config PATH")

    if not args.config:
        parser.error("--config PATH is required (unless using --help-set)")

    cfg = load_cfg(args.config, overrides=args.set)

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        sys.exit(0)

    run_name = cfg["run"]["name"]
    logs_dir = cfg["run"]["logs_dir"]
    run_dir = Path(logs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(run_dir))
    print_config_summary(cfg)
    logger.info(f"Starting NAS experiment: {run_name}\n")

    exp_cfg = build_experiment(cfg)
    exp_logger = Logger(log_dir=str(run_dir), run_name=run_name)

    gpu_ids = cfg["eval"].get("gpu_ids", None)
    weight_pool = WeightPool() if exp_cfg.evolution.use_weight_inheritance else None

    backend = Backend(
        exp_cfg=exp_cfg,
        max_workers=exp_cfg.evolution.num_workers,
        gpu_ids=gpu_ids,
    )

    engine = EvolutionEngine(
        exp_cfg=exp_cfg,
        backend=backend,
        logger_=exp_logger,
        checkpoint_dir=cfg["run"]["checkpoints_dir"],
    )

    exp_logger.save_config_copy(args.config)
    exp_logger.save_resolved_config(cfg)

    try:
        final_pop, best_indiv = engine.run()
    finally:
        backend.shutdown(wait=True)
        logger.info("[Run] Backend shutdown complete.")

    if best_indiv is None:
        raise RuntimeError("No best individual found (no completed evaluations?)")

    individuals = sorted(
        [ind for ind in final_pop.get_all() if ind.metrics is not None],
        key=lambda ind: ind.fitness,
    )

    logger.info("Final population (sorted by fitness):")
    for i, ind in enumerate(individuals):
        depth = sum(len(st.blocks) for st in ind.genome.stages)
        logger.info(f"Rank {i+1}: id={ind.id}")
        logger.info(f"  mse={ind.metrics['mse']:.4f}, params={ind.metrics['params']:.0f}")
        logger.info(f"  family={ind.genome.family}, depth={depth}")

    extra_steps = cfg.get("eval", {}).get("extra_training_steps", 2000)
    test_metrics, finetuned_state, meta = finetune_and_test(
        best_fitness=best_indiv.fitness,
        genome=best_indiv.genome,
        exp_cfg=exp_cfg,
        initial_state_dict_cpu=None,
        extra_training_steps=extra_steps,
    )

    torch.save(
        {
            "genome": best_indiv.genome.to_dict(),
            "model_state_dict": finetuned_state,
            "meta": meta,
            "test_metrics": test_metrics,
        },
        run_dir / f"{run_name}__best_finetuned.pt"
    )
    best_indiv.genome.dump_structure(run_dir / f"{run_name}__best_finetuned.genome.json")
    logger.info(f"[Final] test_metrics={test_metrics}")

    plot_run(
        run_dir,
        exp_cfg=exp_cfg,
        also_plot_predictions=True,
        best_kind="best_finetuned",
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()