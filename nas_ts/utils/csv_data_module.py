from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from pathlib import Path
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ..core.config import CSVDataConfig


def _infer_format(path: str, cfg: CSVDataConfig) -> str:
    fmt = (cfg.file_format or "auto").lower()
    if fmt != "auto":
        return fmt
    suf = Path(path).suffix.lower()
    if suf == ".npz":
        return "npz"
    if suf == ".txt":
        return "txt"
    return "csv"


def _read_tabular_file(path: str, cfg: CSVDataConfig) -> pd.DataFrame:
    """
    Returns a DataFrame with numeric columns (and possibly a date column).
    Supports CSV/TXT via pandas and NPZ via numpy.
    """
    fmt = _infer_format(path, cfg)

    if fmt in ("csv", "txt"):
        df = pd.read_csv(
            path,
            header=None if cfg.has_header is False else "infer",
            sep=cfg.delimiter,
            engine="python",
        )
        return df

    if fmt == "npz":
        with np.load(path, allow_pickle=True) as z:
            key = cfg.npz_key
            if key is None:
                keys = list(z.keys())
                if not keys:
                    raise ValueError(f"NPZ file has no arrays: {path}")
                key = keys[0]
            arr = z[key]

            if arr.ndim == 2:
                pass
            elif arr.ndim == 3:
                if arr.shape[-1] == 1:
                    arr = arr[..., 0]
                elif arr.shape[1] == 1:
                    arr = arr[:, 0, :]
                else:
                    T0, A1, A2 = arr.shape
                    if T0 >= max(A1, A2) and T0 > 4:
                        arr = arr.reshape(T0, A1 * A2)
                    else:
                        arr = arr[0]
            else:
                raise ValueError(f"Expected 2D or 3D array in npz[{key}] but got shape={arr.shape}")

            return pd.DataFrame(arr)

    raise ValueError(f"Unsupported file_format='{fmt}' for path={path}")


def _load_and_stack_tables(paths: Sequence[str], cfg: CSVDataConfig) -> pd.DataFrame:
    """Load one or more files and horizontally concat columns."""
    dfs = [_read_tabular_file(p, cfg) for p in paths]
    min_len = min(len(df) for df in dfs)
    dfs = [df.iloc[:min_len].reset_index(drop=True) for df in dfs]
    return pd.concat(dfs, axis=1)


def _parse_col_spec(cols: Optional[Sequence[str]], df: pd.DataFrame, cfg: CSVDataConfig) -> Optional[list]:
    """
    Convert feature_cols/target_cols into actual DataFrame column labels.
    If cfg.use_col_indices=True, cols are treated as integer indices.
    Otherwise they are treated as column names.
    """
    if cols is None:
        return None

    if cfg.use_col_indices:
        out = []
        for c in cols:
            idx = int(c)
            if idx < 0 or idx >= len(df.columns):
                raise ValueError(f"Column index {idx} out of range for df with {len(df.columns)} cols")
            out.append(df.columns[idx])
        return out

    cols = np.array(cols, dtype=str)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}. Available: {list(df.columns)[:20]}...")
    return list(cols)


def _resolve_date_series(df: pd.DataFrame, cfg: CSVDataConfig) -> Optional[pd.Series]:
    """
    Resolve the datetime series used to create time marks.
    Supports date_col as a column name, integer index, or numeric string index.
    Returns None if not found or disabled.
    """
    date_col = getattr(cfg, "date_col", None)
    if date_col is None:
        return None

    if isinstance(date_col, int):
        return df.iloc[:, date_col] if 0 <= date_col < len(df.columns) else None

    if isinstance(date_col, str):
        s = date_col.strip()
        if not s:
            return None
        if s in df.columns:
            return df[s]
        try:
            idx = int(s)
        except ValueError:
            return None
        return df.iloc[:, idx] if 0 <= idx < len(df.columns) else None

    return None


def _build_time_marks_from_date(date_series: pd.Series) -> torch.Tensor:
    """
    Build timeF-style marks from a datetime column.
    Output shape: [T, 4] — (month, day, weekday, hour) scaled to ~[0, 1].
    """
    dt = pd.to_datetime(date_series)
    marks = torch.stack([
        torch.tensor(dt.dt.month.values, dtype=torch.float32) / 12.0,
        torch.tensor(dt.dt.day.values, dtype=torch.float32) / 31.0,
        torch.tensor(dt.dt.weekday.values, dtype=torch.float32) / 6.0,
        torch.tensor(dt.dt.hour.values, dtype=torch.float32) / 23.0,
    ], dim=1)
    return marks


class TimeSeriesWindowDataset(Dataset):
    """
    Sliding-window dataset with optional time marks.
    Each sample: (X [L_in, D_in], Y [L_out, D_out], X_mark [L_in, C], Y_mark [L_out, C])
    """
    def __init__(
        self,
        series: torch.Tensor,
        marks: Optional[torch.Tensor],
        L_in: int,
        L_out: int,
        feature_idx: Optional[torch.Tensor] = None,
        target_idx: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.series = series
        self.marks = marks
        self.L_in = int(L_in)
        self.L_out = int(L_out)

        D_all = series.size(1)
        self.feature_idx = feature_idx if feature_idx is not None else torch.arange(D_all)
        self.target_idx = target_idx if target_idx is not None else self.feature_idx

        T = series.size(0)
        self.num_samples = T - (self.L_in + self.L_out) + 1
        if self.num_samples <= 0:
            raise ValueError(f"Series too short: T={T}, L_in={L_in}, L_out={L_out}")

        if self.marks is not None and self.marks.size(0) != T:
            raise ValueError(f"marks length mismatch: marks T={self.marks.size(0)} vs series T={T}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x_start = idx
        x_end = idx + self.L_in
        y_start = x_end
        y_end = y_start + self.L_out

        X = self.series[x_start:x_end][:, self.feature_idx]
        Y = self.series[y_start:y_end][:, self.target_idx]

        if self.marks is None:
            X_mark = torch.empty(self.L_in, 0, dtype=torch.float32)
            Y_mark = torch.empty(self.L_out, 0, dtype=torch.float32)
        else:
            X_mark = self.marks[x_start:x_end]
            Y_mark = self.marks[y_start:y_end]

        return X, Y, X_mark, Y_mark


def make_csv_dataloaders(cfg: CSVDataConfig):
    """
    Loads CSV(s), builds time marks from date column if present,
    normalizes on train split only, splits chronologically,
    and returns (train_loader, val_loader, test_loader, meta).
    """
    df = _load_and_stack_tables(cfg.paths, cfg)

    # Resolve and remove date column before building numeric tensor
    date_series = _resolve_date_series(df, cfg)
    if date_series is not None:
        date_col = getattr(cfg, "date_col", None)
        if isinstance(date_col, str) and date_col in df.columns:
            df = df.drop(columns=[date_col])
        elif isinstance(date_col, int) and 0 <= date_col < len(df.columns):
            df = df.drop(columns=[df.columns[date_col]])

    marks_full = _build_time_marks_from_date(date_series) if date_series is not None else None

    feat_cols_resolved = _parse_col_spec(cfg.feature_cols, df, cfg)
    tgt_cols_resolved = _parse_col_spec(cfg.target_cols, df, cfg)

    df_feat = df[feat_cols_resolved] if feat_cols_resolved is not None else df.select_dtypes(include=["number"])
    df_tgt = df[tgt_cols_resolved] if tgt_cols_resolved is not None else df_feat

    feat_cols = list(df_feat.columns)
    tgt_cols = list(df_tgt.columns)

    # Union of needed columns so target_cols can differ from feature_cols
    all_cols = list(dict.fromkeys(feat_cols + tgt_cols))
    df_all = df[all_cols].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    values = torch.tensor(df_all.values, dtype=torch.float32)
    T, D_all = values.shape

    col_to_idx = {c: i for i, c in enumerate(df_all.columns)}
    feature_idx = torch.tensor([col_to_idx[c] for c in feat_cols], dtype=torch.long)
    target_idx = torch.tensor([col_to_idx[c] for c in tgt_cols], dtype=torch.long)

    L_in = int(cfg.input_length)
    L_out = int(cfg.pred_length)

    t_train_end = max(int(T * cfg.train_ratio), L_in + L_out)
    t_val_end = max(int(T * (cfg.train_ratio + cfg.val_ratio)), t_train_end + L_out)
    t_train_end = min(t_train_end, T)
    t_val_end = min(t_val_end, T)

    if cfg.normalize:
        train_raw = values[:t_train_end]
        mean = train_raw.mean(dim=0, keepdim=True)
        std = train_raw.std(dim=0, keepdim=True) + 1e-6
        values = (values - mean) / std

    def _slice_with_context(arr: Optional[torch.Tensor], start: int, end: int) -> Optional[torch.Tensor]:
        if arr is None:
            return None
        return arr[max(0, start - L_in):end]

    train_series = values[:t_train_end]
    val_series = values[max(0, t_train_end - L_in):t_val_end]
    test_series = values[max(0, t_val_end - L_in):]

    train_marks = marks_full[:t_train_end] if marks_full is not None else None
    val_marks = _slice_with_context(marks_full, t_train_end, t_val_end)
    test_marks = _slice_with_context(marks_full, t_val_end, T)

    train_ds = TimeSeriesWindowDataset(train_series, train_marks, L_in, L_out, feature_idx, target_idx)
    val_ds = TimeSeriesWindowDataset(val_series, val_marks, L_in, L_out, feature_idx, target_idx)

    test_ds: Dataset
    if test_series.size(0) >= (L_in + L_out):
        test_ds = TimeSeriesWindowDataset(test_series, test_marks, L_in, L_out, feature_idx, target_idx)
    else:
        # Always return a test_loader even if the test split is too short
        empty_X = torch.empty(0, L_in, len(feature_idx), dtype=torch.float32)
        empty_Y = torch.empty(0, L_out, len(target_idx), dtype=torch.float32)
        empty_Xm = torch.empty(0, L_in, 0, dtype=torch.float32)
        empty_Ym = torch.empty(0, L_out, 0, dtype=torch.float32)
        test_ds = TensorDataset(empty_X, empty_Y, empty_Xm, empty_Ym)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    meta: Dict[str, Any] = {
        "n_raw_timesteps": int(T),
        "d_all": int(D_all),
        "d_in": int(len(feature_idx)),
        "d_out": int(len(target_idx)),
        "mark_dim": int(marks_full.size(1)) if marks_full is not None else 0,
        "feature_cols": feat_cols,
        "target_cols": tgt_cols,
        "all_cols": all_cols,
        "input_length": int(L_in),
        "pred_length": int(L_out),
        "t_train_end": int(t_train_end),
        "t_val_end": int(t_val_end),
        "n_train_windows": int(len(train_ds)),
        "n_val_windows": int(len(val_ds)),
        "n_test_windows": int(len(test_ds)),
        "normalize": bool(cfg.normalize),
        "has_date_marks": bool(marks_full is not None),
    }
    if cfg.normalize:
        meta["global_mean"] = mean
        meta["global_std"] = std

    return train_loader, val_loader, test_loader, meta