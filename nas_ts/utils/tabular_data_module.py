import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# "clean"  our protocol: three disjoint splits, scaler fitted on train only.
# "exaqc"  reproduces EXAQC's protocol for a like-for-like comparison.
SPLIT_MODES = ("clean", "exaqc")

# EXAQC uses a single stratified 80/20 split and scales features to [0, pi]
# (MinMaxScaler * pi) fitted on the whole dataset before splitting.
EXAQC_HOLDOUT_FRACTION = 0.2
EXAQC_FEATURE_SCALE = math.pi

# The exaqc warning is per process, not per evaluation: loaders are rebuilt for
# every genome, and 500 copies of the same caveat would bury the run log.
_EXAQC_WARNED = False


@dataclass
class TabularDataConfig:
    path: str
    feature_cols: Optional[List[str]] = None   # None => all columns except target_col
    target_col: str = "target"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 0
    normalize: bool = True
    random_seed: int = 42
    split_mode: str = "clean"


class TabularClassificationDataset(Dataset):
    """
    Each sample: (X [1, N_features], y [scalar long], X_mark [1, 0], Y_mark [1, 0])
    The leading '1' is a seq_len=1 axis so existing seq-shaped tokenizers (e.g. var/
    iTransformer) work unmodified. Empty mark tensors keep the 4-tuple batch format
    used by _unpack_batch in evaluate.py, without implying any temporal structure.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 2, f"expected [N_samples, N_features], got {X.shape}"
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx: int):
        x = self.X[idx].unsqueeze(0)          # [1, N_features]
        y = self.y[idx]                       # scalar long
        x_mark = torch.empty(1, 0, dtype=torch.float32)
        y_mark = torch.empty(1, 0, dtype=torch.float32)
        return x, y, x_mark, y_mark


def _warn_exaqc_protocol_once(n_holdout: int) -> None:
    global _EXAQC_WARNED
    if _EXAQC_WARNED:
        return
    _EXAQC_WARNED = True
    logger.warning(
        f"[Split] split_mode='exaqc': the same {n_holdout} samples serve as BOTH "
        f"validation and test. Model selection and early stopping run on them, so "
        f"the reported test accuracy is a fitted number, NOT a held-out estimate, "
        f"and is optimistically biased. Features are also MinMax-scaled to "
        f"[0, pi] on the FULL dataset before splitting, which leaks the holdout's "
        f"range into training. This mode exists only to compare like-for-like "
        f"with EXAQC's published protocol; use split_mode='clean' for any number "
        f"presented as a generalisation estimate."
    )


def _split_clean(X_all, y_idx, cfg):
    """
    Three disjoint stratified splits; z-score fitted on train only.

    test_ratio is whatever train_ratio and val_ratio leave over.
    """
    test_ratio = 1.0 - cfg.train_ratio - cfg.val_ratio
    if test_ratio <= 0:
        raise ValueError(f"train_ratio + val_ratio must be < 1.0, got "
                          f"{cfg.train_ratio} + {cfg.val_ratio}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_idx, test_size=test_ratio, stratify=y_idx, random_state=cfg.random_seed
    )
    val_frac_of_trainval = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac_of_trainval,
        stratify=y_trainval, random_state=cfg.random_seed
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    mean = std = None
    if cfg.normalize:
        # Fitted on train only: val and test must not inform the scaling.
        mean = X_train_t.mean(dim=0, keepdim=True)
        std = X_train_t.std(dim=0, keepdim=True) + 1e-6
        X_train_t = (X_train_t - mean) / std
        X_val_t = (X_val_t - mean) / std
        X_test_t = (X_test_t - mean) / std

    splits = (X_train_t, torch.tensor(y_train, dtype=torch.long),
              X_val_t, torch.tensor(y_val, dtype=torch.long),
              X_test_t, torch.tensor(y_test, dtype=torch.long))
    scaling = {"scaling": "zscore_train_only", "global_mean": mean, "global_std": std}
    return splits, scaling


def _split_exaqc(X_all, y_idx, cfg):
    """
    EXAQC's protocol, reproduced: one stratified 80/20 split, MinMaxScaler * pi
    fitted on the FULL dataset before splitting, and the 20% used as both
    validation and test -- exactly as in their code. train_ratio and val_ratio
    are ignored here; the fraction is fixed by their protocol.
    """
    X_scaled = MinMaxScaler().fit_transform(X_all) * EXAQC_FEATURE_SCALE

    X_train, X_hold, y_train, y_hold = train_test_split(
        X_scaled, y_idx, test_size=EXAQC_HOLDOUT_FRACTION,
        stratify=y_idx, random_state=cfg.random_seed
    )

    _warn_exaqc_protocol_once(len(y_hold))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_hold_t = torch.tensor(X_hold, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_hold_t = torch.tensor(y_hold, dtype=torch.long)

    # val and test are the same samples, deliberately. Cloned so a downstream
    # in-place edit to one cannot silently alter the other.
    splits = (X_train_t, y_train_t,
              X_hold_t, y_hold_t,
              X_hold_t.clone(), y_hold_t.clone())
    scaling = {"scaling": f"minmax_x_pi_full_dataset", "global_mean": None, "global_std": None}
    return splits, scaling


def make_tabular_dataloaders(cfg: TabularDataConfig):
    """
    Loads a tabular CSV, splits it according to cfg.split_mode, scales features,
    and returns (train_loader, val_loader, test_loader, meta) — matching
    make_csv_dataloaders' return signature.

    split_mode:
      "clean"  stratified train/val/test from the configured ratios, z-score
               fitted on train only. Model selection on val, one final
               evaluation on test.
      "exaqc"  EXAQC's published protocol: stratified 80/20, MinMax * pi fitted
               on the full dataset, and the 20% used as both val and test.
    """
    df = pd.read_csv(cfg.path)

    if cfg.target_col not in df.columns:
        raise ValueError(f"target_col={cfg.target_col!r} not found in {cfg.path}. "
                          f"Available: {list(df.columns)}")

    feat_cols = cfg.feature_cols or [c for c in df.columns if c != cfg.target_col]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df[cfg.target_col].to_numpy()

    classes = sorted(pd.unique(y_all).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[v] for v in y_all], dtype=np.int64)

    n = len(df)

    split_mode = str(cfg.split_mode or "clean").lower()
    if split_mode not in SPLIT_MODES:
        raise ValueError(
            f"Unknown data.tabular.split_mode={cfg.split_mode!r}. "
            f"Valid modes: {', '.join(SPLIT_MODES)}."
        )

    if split_mode == "exaqc":
        splits, scaling = _split_exaqc(X_all, y_idx, cfg)
    else:
        splits, scaling = _split_clean(X_all, y_idx, cfg)

    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t = splits

    train_ds = TabularClassificationDataset(X_train_t, y_train_t)
    val_ds = TabularClassificationDataset(X_val_t, y_val_t)
    test_ds = TabularClassificationDataset(X_test_t, y_test_t)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    meta: Dict[str, Any] = {
        "n_samples": int(n),
        "d_in": int(len(feat_cols)),
        "d_out": int(len(feat_cols)),   # unused for classification; kept for interface parity
        "num_classes": int(len(classes)),
        "class_labels": classes,
        "feature_cols": feat_cols,
        "target_col": cfg.target_col,
        "input_length": 1,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "normalize": bool(cfg.normalize),
        "split_mode": split_mode,
        # The seed the split actually used, so a results record never has to
        # guess it from a config that may have left it defaulted.
        "random_seed": int(cfg.random_seed),
        "scaling": scaling["scaling"],
        # True when the test split IS the validation split, so any consumer of
        # these results can see that the test number is not held out.
        "val_is_test": bool(split_mode == "exaqc"),
    }
    if scaling["global_mean"] is not None:
        meta["global_mean"] = scaling["global_mean"]
        meta["global_std"] = scaling["global_std"]

    return train_loader, val_loader, test_loader, meta
