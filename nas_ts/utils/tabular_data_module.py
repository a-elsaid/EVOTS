from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


@dataclass
class TabularDataConfig:
    path: str
    feature_cols: Optional[List[str]] = None   # None => all columns except target_col
    target_col: str = "target"
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    batch_size: int = 32
    num_workers: int = 0
    normalize: bool = True
    random_seed: int = 42


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


def make_tabular_dataloaders(cfg: TabularDataConfig):
    """
    Loads a tabular CSV, does a stratified train/val/test split, standardizes
    features on the train split only, and returns (train_loader, val_loader,
    test_loader, meta) — matching make_csv_dataloaders' return signature.
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
    test_ratio = 1.0 - cfg.train_ratio - cfg.val_ratio
    if test_ratio <= 0:
        raise ValueError(f"train_ratio + val_ratio must be < 1.0, got "
                          f"{cfg.train_ratio} + {cfg.val_ratio}")

    # Stratified split: first carve off test, then split remainder into train/val
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
        mean = X_train_t.mean(dim=0, keepdim=True)
        std = X_train_t.std(dim=0, keepdim=True) + 1e-6
        X_train_t = (X_train_t - mean) / std
        X_val_t = (X_val_t - mean) / std
        X_test_t = (X_test_t - mean) / std

    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

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
    }
    if cfg.normalize:
        meta["global_mean"] = mean
        meta["global_std"] = std

    return train_loader, val_loader, test_loader, meta
