import yaml
from typing import Optional

from .config import (
    RunInfo, TaskConfig, DatasetConfig, EvalConfig,
    SelectionConfig, SearchSpaceConfig,
    EvolutionConfig, GenomeConstraints, ExperimentConfig
)
from ..utils.csv_data_module import make_csv_dataloaders, CSVDataConfig
from ..utils.tabular_data_module import make_tabular_dataloaders, TabularDataConfig
from ..search.seeds import make_iTransformer_like, make_PatchTST_like


def _parse_value(s: str):
    return yaml.safe_load(s)


def _set_by_dots(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"Override key path '{dotted_key}' not found in config.")
        cur = cur[k]

    # special case: always parse as list
    list_keys = {
        "metrics", "paths", "families", "objectives", "feature_cols", "target_cols",
        "stage_tokenizers", "stage_count_range", "data_types", "block_types",
        "var_encoder_types", "var_conv_kernels", "var_fft_keep_ratios", "var_decomp_kernels",
        "cross_groups", "cross_patch_sizes", "cross_strides", "cross_encoder_types",
        "cross_conv_kernels", "cross_pools",
        "quantum_nlayers_range", "quantum_entangle_patterns", "quantum_gate_sets", "quantum_use_ffn_options",
        "depth_range", "model_dim_range",
        "num_heads_range", "ff_mult_range", "patch_sizes", "strides", "overlaps",
        "per_channel_options", "cross_fusions", "freq_types", "freq_keep_ratios",
        "decomp_modes", "decomp_kernel_sizes", "conv_kernel_sizes", "conv_dilations",
        "pos_encoding_options", "gpu_ids",
    }

    if keys[-1] in list_keys:
        items = [v.strip() for v in value.split(",") if v.strip()]
        parsed_items = []
        for it in items:
            # gpu ids should become ints when possible
            try:
                parsed_items.append(int(it))
            except ValueError:
                parsed_items.append(it)
        cur[keys[-1]] = parsed_items
    else:
        cur[keys[-1]] = value


def load_cfg(path: str, overrides: Optional[list[str]]) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}'. Use key.path=value")
        k, v = item.split("=", 1)
        _set_by_dots(cfg, k, _parse_value(v))

    return cfg


def aggregate_mse(metrics: dict) -> float:
    return metrics["mse"]


def aggregate_classification_fitness(metrics: dict) -> float:
    # Every comparison in engine.py/selection_ops.py assumes "lower fitness is
    # better" (tournament selection, early stopping, Pareto dominance). Using
    # loss (not raw accuracy) as fitness keeps that assumption true without
    # touching any of those comparison sites.
    return metrics["loss"]


def build_experiment(cfg: dict) -> ExperimentConfig:
    task = TaskConfig(**cfg["task"])

    if task.task_type == "classification":
        # WindowNorm degenerates to all-zero output at input_length=1 (mean of a
        # single point is itself, so x - mean = 0 everywhere). Classification
        # tabular data uses input_length=1, so this must stay off.
        if task.use_norm:
            raise ValueError(
                "task.use_norm must be false for task_type='classification' "
                "(WindowNorm divides by ~0 at input_length=1 and silently "
                "zeroes every feature)."
            )

        tab_cfg = cfg["data"]["tabular"]
        tab_dm_cfg = TabularDataConfig(
            path=tab_cfg["path"],
            feature_cols=tab_cfg.get("feature_cols"),
            target_col=tab_cfg.get("target_col", "target"),
            train_ratio=tab_cfg["train_ratio"],
            val_ratio=tab_cfg["val_ratio"],
            batch_size=tab_cfg["batch_size"],
            num_workers=tab_cfg.get("num_workers", 0),
            normalize=tab_cfg.get("normalize", True),
            random_seed=tab_cfg.get("random_seed", 42),
        )
        ds_cfg = DatasetConfig(
            name="Tabular",
            loader_fn=make_tabular_dataloaders,
            loader_kwargs={"cfg": tab_dm_cfg},
            train_split=None,
            val_split=None,
        )
        batch_size = tab_dm_cfg.batch_size
        aggregation_fn = aggregate_classification_fitness
    else:
        csv_dm_cfg = CSVDataConfig(
            paths=cfg["data"]["csv"]["paths"],
            input_length=task.input_length,
            pred_length=task.pred_length,
            feature_cols=cfg["data"]["csv"].get("feature_cols"),
            target_cols=cfg["data"]["csv"].get("target_cols"),
            train_ratio=cfg["data"]["csv"]["train_ratio"],
            val_ratio=cfg["data"]["csv"]["val_ratio"],
            batch_size=cfg["data"]["csv"]["batch_size"],
            num_workers=cfg["data"]["csv"]["num_workers"],
            normalize=cfg["data"]["csv"]["normalize"],
            file_format=cfg["data"]["csv"].get("file_format", "auto"),
            has_header=cfg["data"]["csv"].get("has_header", True),
            delimiter=cfg["data"]["csv"].get("delimiter", None),
            npz_key=cfg["data"]["csv"].get("npz_key", None),
            use_col_indices=cfg["data"]["csv"].get("use_col_indices", False),
            date_col=cfg["data"]["csv"].get("date_col", "date"),
        )
        ds_cfg = DatasetConfig(
            name="CSV",
            loader_fn=make_csv_dataloaders,
            loader_kwargs={"cfg": csv_dm_cfg},
            train_split=None,
            val_split=None,
        )
        batch_size = csv_dm_cfg.batch_size
        aggregation_fn = aggregate_mse

    eval_cfg = EvalConfig(
        task=task,
        datasets=[ds_cfg],
        training_steps=cfg["eval"]["training_steps"],
        extra_training_steps=cfg["eval"]["extra_training_steps"],
        early_stopping=cfg["eval"]["early_stopping"],
        early_stop_patience=cfg["eval"]["early_stop_patience"],
        early_stop_min_delta=cfg["eval"]["early_stop_min_delta"],
        early_stop_checks=cfg["eval"]["early_stop_checks"],
        batch_size=batch_size,
        optimizer=cfg["eval"]["optimizer"],
        optimizer_kwargs=cfg["eval"].get("optimizer_kwargs", {}),
        device=cfg["eval"]["device"],
    )

    sel_cfg = SelectionConfig(
        mode=cfg["selection"]["mode"],
        objectives=cfg["selection"]["objectives"],
        aggregation=aggregation_fn,
    )

    # Filter unknown keys to avoid crashes on config fields not in the dataclass
    ss_dict = {k: v for k, v in cfg["search_space"].items() if k in SearchSpaceConfig.__dataclass_fields__}
    search_space = SearchSpaceConfig(**ss_dict)

    evo_cfg = EvolutionConfig(**cfg["evo"])
    constraints = GenomeConstraints(**cfg["constraints"])

    seed_genomes = []
    for seed in cfg.get("seeds", []):
        if seed["type"].lower() in ("itransformer", "it"):
            seed_genomes.append(make_iTransformer_like(model_dim=seed["model_dim"], num_layers=seed["num_layers"]))
        elif seed["type"].lower() in ("patchtst", "patch"):
            seed_genomes.append(make_PatchTST_like(model_dim=seed["model_dim"], num_layers=seed["num_layers"]))
        else:
            raise ValueError(f"Unknown seed type: {seed['type']}")

    checkpoint_dir = (
        cfg["run"].get("checkpoint_dir")
        or cfg["run"].get("checkpoints_dir")
        or None
    )

    run_info = RunInfo(
        name=cfg["run"]["name"],
        logs_dir=cfg["run"]["logs_dir"],
        checkpoint_dir=checkpoint_dir,
        notes=cfg["run"].get("notes", None),
    )

    return ExperimentConfig(
        run_info=run_info,
        eval_config=eval_cfg,
        selection_config=sel_cfg,
        search_space=search_space,
        evolution=evo_cfg,
        genome_constraints=constraints,
        seed_genomes=seed_genomes,
    )