# nas_ts/config.py
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Sequence, Tuple, Any

from .genome_v2 import Genome


# ---- Problem / task side ----

@dataclass
class RunInfo:
    name: str
    logs_dir: str
    checkpoint_dir: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class TaskConfig:
    task_type: str
    input_length: int
    pred_length: int
    d_in: int
    d_out: int
    metrics: List[str]
    use_norm: bool = True


@dataclass
class DatasetConfig:
    name: str
    loader_fn: Callable[..., Any]
    loader_kwargs: Dict[str, Any]
    train_split: Optional[float] = None
    val_split: Optional[float] = None


@dataclass
class CSVDataConfig:
    paths: Sequence[str]
    input_length: int
    pred_length: int
    feature_cols: Optional[Sequence[str]] = None
    target_cols: Optional[Sequence[str]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    max_rows: Optional[int] = None  # clip series to this many rows before splitting (e.g. 14400 for ETTh paper protocol)
    batch_size: int = 32
    num_workers: int = 1
    normalize: bool = True
    file_format: str = "auto"         # "auto" | "csv" | "txt" | "npz"
    has_header: bool = True           # if False, treat file as headerless
    delimiter: Optional[str] = None   # for csv/txt (None => pandas default)
    npz_key: Optional[str] = None     # which array key inside npz (None => first array found)
    # If True: interpret feature_cols/target_cols as integer indices (e.g. ["0","1","2"])
    # If False: interpret as column names (e.g. ["HUFL","HULL",...])
    use_col_indices: bool = False
    # If file has a date column and you want marks, set date_col.
    # - header files: "date"
    # - headerless files: use_col_indices=True and date_col="0" (or set it to the index as string)
    date_col: Optional[str] = "date"


@dataclass
class EvalConfig:
    task: TaskConfig
    datasets: List[DatasetConfig]
    training_steps: int
    extra_training_steps: int
    early_stopping: bool
    early_stop_checks: int
    early_stop_min_delta: float
    early_stop_patience: int
    batch_size: int
    optimizer: str
    optimizer_kwargs: Dict[str, Any]
    device: str = "auto"
    # num_workers: int = 1


# ---- Selection ----

@dataclass
class SelectionConfig:
    mode: str
    objectives: List[str]
    aggregation: Optional[Callable[[Dict[str, float]], float]] = None
    use_pareto: bool = True


# ---- Search space & evolution ----

@dataclass
class SearchSpaceConfig:
    families: List[str]

    depth_range: Tuple[int, int]
    model_dim_range: Tuple[int, int]
    num_heads_range: Tuple[int, int]
    ff_mult_range: Tuple[float, float]

    patch_sizes: List[int]
    strides: List[int]
    overlaps: List[float]
    per_channel_options: List[bool]

    cross_groups: List[int]
    cross_fusions: List[str]

    freq_types: List[str]
    freq_keep_ratios: List[float]

    decomp_modes: List[str]
    decomp_kernel_sizes: List[int]

    conv_kernel_sizes: List[int]
    conv_dilations: List[int]

    block_types: List[str]

    pos_encoding_options: List[str]
    dropout_range: Tuple[float, float]

    # ---- VarToken head search knobs (for tokenizer="var") ----
    var_encoder_types: List[str] = field(default_factory=lambda: ["linear", "conv", "fft", "decomp_linear"])
    var_conv_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])
    var_fft_keep_ratios: List[float] = field(default_factory=lambda: [0.125, 0.25, 0.5])
    var_decomp_kernels: List[int] = field(default_factory=lambda: [3, 5, 7])

    # ---- Quantum block knobs ----
    quantum_nlayers_range: Tuple[int, int] = (1, 3)
    quantum_entangle_patterns: List[str] = field(default_factory=lambda: ["linear", "circular"])
    quantum_gate_sets: List[str] = field(default_factory=lambda: ["rx_ry", "rx_ry_rz"])
    quantum_use_ffn_options: List[bool] = field(default_factory=lambda: [True, False])

    # Qubit ranges set from measurement, not guesswork. Simulation cost is 2^n
    # complex amplitudes per token, vmapped over B*N tokens, so an over-wide range
    # makes workers run out of memory. A Python-level allocation failure is caught
    # and scores that genome inf, indistinguishable from a genuinely bad
    # architecture; a native OOM kills the worker, breaks the process pool and
    # takes down the whole run, which the suite then records as failed.
    #
    # MEASURED ON CPU (Apple Silicon laptop), at breast-cancer scale: batch 32,
    # 30 tokens per sample, d_model 256, so 960 tokens per vmapped call — the
    # widest of the four classification datasets. Figures are one full training
    # evaluation (200 epochs x 13 batches) with a single quantum block in the
    # model, against a classical genome of the same shape:
    #
    #   amplitude  n=8   21.5 min  1.1 GB   0.9x classical
    #   amplitude  n=12  42.4 min  1.8 GB   1.7x        (expval_z: 62 min, 2.4 GB)
    #   amplitude  n=14 145.8 min  3.8 GB   5.9x        (expval_z: 316 min, 8.6 GB)
    #   angle      n=12  22.1 min  1.3 GB   0.9x
    #   angle      n=14  26.5 min  2.1 GB   1.1x        (expval_z: 138 min, 7.7 GB)
    #
    # Amplitude stops at 12: at 14 it costs 6-13x a classical genome and peaks at
    # 3.8-8.6 GB per worker, which is 42-95 GB across the 11 workers the configs
    # request. Angle reaches 14 for ~1x, because its input projection is
    # d_model -> n rather than d_model -> 2^n; that asymmetry is why the two
    # encodings get separate ranges rather than one shared one.
    #
    # The cluster runs on GPU, where the arithmetic is far cheaper relative to
    # memory traffic, so these ratios will not carry over directly — expect the
    # wall-clock multipliers to shrink and the memory ceiling to bind sooner (GPU
    # RAM per worker is smaller than host RAM). Re-measure with
    # tools/bench_quantum.py --device cuda before trusting either bound there.
    quantum_encodings: List[str] = field(default_factory=lambda: ["amplitude", "angle"])
    quantum_readouts: List[str] = field(default_factory=lambda: ["state", "prob", "expval_z"])
    quantum_amplitude_qubits_range: Tuple[int, int] = (4, 12)
    quantum_angle_qubits_range: Tuple[int, int] = (4, 14)
    quantum_reupload_options: List[bool] = field(default_factory=lambda: [True, False])

    # ---- Stages (for genome_v2) ----
    stage_count_range: Tuple[int, int] = (1, 3)
    stage_tokenizers: List[str] = field(default_factory=lambda: ["time", "var", "patch", "cross"])
    # No longer searched: retokenize is position-determined (stage0="none", i>=1="cross_attn") per issue #3.
    stage_retokens: List[str] = field(default_factory=lambda: ["none", "cross_attn"])

    # ---- CrossToken head search knobs (for tokenizer="cross") ----
    cross_patch_sizes: List[int] = field(default_factory=lambda: [4, 8])
    cross_strides: List[int] = field(default_factory=lambda: [1, 2])
    cross_encoder_types: List[str] = field(default_factory=lambda: ["linear", "conv"])
    cross_conv_kernels: List[int] = field(default_factory=lambda: [3, 5])
    cross_pools: List[str] = field(default_factory=lambda: ["avg", "max"])


@dataclass
class EvolutionConfig:
    population_size: int
    max_evals: int
    init_seed_fraction: float
    mutation_rate: float
    crossover_rate: float
    steady_state: bool = True
    tournament_size: int = 3
    use_weight_inheritance: bool = True
    evals_per_generation: int = 20
    random_seed: int = 42
    num_workers: int = 10
    gpu_ids: Optional[List[int]] = None


@dataclass
class GenomeConstraints:
    min_blocks: int = 2
    max_blocks: int = 12
    max_conv_blocks: int = 4
    max_freq_blocks: int = 4
    max_cross_dim_blocks: int = 4

    # Floor on quantum blocks per genome. 0 (the default) leaves the search free
    # to reject quantum entirely, which is what every existing config does.
    # Setting it to 1 or more forces every genome to contain a circuit, turning
    # "is a circuit useful here" into "what is the best circuit-containing
    # architecture" -- a different question, so it gets its own config directory
    # rather than changing the free condition.
    min_quantum_blocks: int = 0


@dataclass
class ExperimentConfig:
    run_info: RunInfo
    eval_config: EvalConfig
    selection_config: SelectionConfig
    search_space: SearchSpaceConfig
    evolution: EvolutionConfig
    genome_constraints: Optional[GenomeConstraints] = None
    seed_genomes: Optional[List[Genome]] = None