import argparse

HELP_SET_TEXT = (
    "--set KEY=VALUE\n\n"
    "Override config values using dot notation. Can be used multiple times.\n\n"
    "Format:\n"
    "  --set a.b.c=value\n\n"
    "Examples:\n"
    "  --set evo.population_size=12\n"
    "  --set eval.training_steps=100\n"
    "  --set eval.extra_training_steps=2000\n"
    "  --set eval.optimizer_kwargs.lr=0.0005\n"
    "  --set data.csv.batch_size=64\n"
    "  --set data.csv.paths='[\"./ETTh1.csv\"]'\n\n"
    "------------------------------------------------------------\n"
    "CONFIG REFERENCE (required vs defaults)\n"
    "------------------------------------------------------------\n\n"
    "RUN (script-level, not from dataclasses)\n"
    "  run.name                (str, required)\n"
    "  run.logs_dir            (str, required)\n"
    "  run.checkpoints_dir     (str, required)\n\n"
    "TASK (TaskConfig: all required)\n"
    "  task.task_type          (str, required)  e.g. forecasting | classification\n"
    "  task.input_length       (int, required)\n"
    "  task.pred_length        (int, required)\n"
    "  task.d_in               (int, required)\n"
    "  task.d_out              (int, required)\n"
    "  task.metrics            (list[str], required)  e.g. [mse, mae, params]\n\n"
    "DATA.CSV (CSVDataConfig)\n"
    "  data.csv.paths          (list[str], required)\n"
    "  data.csv.feature_cols   (list[str] | null, default=null)\n"
    "  data.csv.target_cols    (list[str] | null, default=null)\n"
    "  data.csv.train_ratio    (float, default=0.7)\n"
    "  data.csv.val_ratio      (float, default=0.1)\n"
    "  data.csv.max_rows       (int | null, default=null)  clip series before splitting; e.g. 14400 for ETTh paper protocol\n"
    "  data.csv.batch_size     (int, default=32)\n"
    "  data.csv.num_workers    (int, default=1)\n"
    "  data.csv.normalize      (bool, default=true)\n\n"
    "EVAL (EvalConfig)\n"
    "  eval.training_steps     (int, required)\n"
    "  eval.optimizer          (str, required)  adam | adamw | ...\n"
    "  eval.optimizer_kwargs   (dict, required) e.g. {lr: 0.001, weight_decay: 0.0001}\n"
    "  eval.device             (str, default=auto)  auto | cpu | cuda | mps\n"
    "  NOTE: batch_size/num_workers are typically taken from data.csv.* in your builder code.\n\n"
    "SELECTION (SelectionConfig)\n"
    "  selection.mode          (str, required)  single | multi\n"
    "  selection.objectives    (list[str], required) e.g. [mse] or [mse, params]\n"
    "  selection.use_pareto    (bool, default=true)\n"
    "  NOTE: selection.aggregation is a Python function set in code (not YAML).\n\n"
    "SEARCH_SPACE (SearchSpaceConfig)\n"
    "  All fields required (no defaults in code). Your YAML must include:\n"
    "    families, depth_range, model_dim_range, num_heads_range, ff_mult_range,\n"
    "    patch_sizes, strides, overlaps, per_channel_options,\n"
    "    cross_groups, cross_fusions,\n"
    "    freq_types, freq_keep_ratios,\n"
    "    decomp_modes, decomp_kernel_sizes,\n"
    "    conv_kernel_sizes, conv_dilations,\n"
    "    block_types, pos_encoding_options, dropout_range\n\n"
    "EVOLUTION (EvolutionConfig)\n"
    "  evo.population_size        (int, required)\n"
    "  evo.max_evals              (int, required)\n"
    "  evo.init_seed_fraction     (float, required)\n"
    "  evo.mutation_rate          (float, required)\n"
    "  evo.crossover_rate         (float, required)\n"
    "  evo.steady_state           (bool, default=true)\n"
    "  evo.tournament_size        (int, default=3)\n"
    "  evo.use_weight_inheritance (bool, default=true)\n"
    "  evo.evals_per_generation   (int, default=20)\n"
    "  evo.random_seed            (int, default=42)\n"
    "  evo.num_workers            (int, default=10)\n\n"
    "CONSTRAINTS (GenomeConstraints)\n"
    "  constraints.min_blocks           (int, default=2)\n"
    "  constraints.max_blocks           (int, default=12)\n"
    "  constraints.max_conv_blocks      (int, default=4)\n"
    "  constraints.max_freq_blocks      (int, default=4)\n"
    "  constraints.max_cross_dim_blocks (int, default=4)\n\n"
    "SEEDS (script-level list used by your builder)\n"
    "  seeds[i].type             (str, optional)  e.g. iTransformer | PatchTST\n"
    "  seeds[i].model_dim        (int, optional)\n"
    "  seeds[i].num_layers       (int, optional)\n\n"
    "Notes:\n"
    "- Values in YOUR YAML are not necessarily code defaults; defaults above come from dataclasses.\n"
    "- For list/dict overrides, pass valid YAML/JSON strings, e.g.:\n"
    "    --set data.csv.paths='[\"./ETTh1.csv\"]'\n"
)

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_exp.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="NAS-TS Experiment Runner\n\nUse --help-set for detailed override help.",
    )

    # NOTE: NOT required here (we enforce manually in main)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (required unless using --help-set).",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (dot notation). See --help-set",
    )

    # accept both spellings
    parser.add_argument(
        "--help-set",
        dest="help_set",
        action="store_true",
        help="Show detailed help for --set overrides and exit.",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved config (requires --config) and exit.",
    )

    return parser