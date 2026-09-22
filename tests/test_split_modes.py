"""
Two split protocols: ours and EXAQC's.

"clean" is the honest one -- three disjoint stratified splits, scaler fitted on
train only, test touched once at the end. "exaqc" reproduces EXAQC's published
protocol so the comparison is like-for-like: a single stratified 80/20, features
MinMax-scaled to [0, pi] on the FULL dataset before splitting, and the 20% used
as both validation and test. Numbers from "exaqc" are fitted, not held out.
"""

import math

import numpy as np
import pytest
import torch

from conftest import require_tabular
from nas_ts.utils import tabular_data_module as tdm
from nas_ts.utils.tabular_data_module import (
    EXAQC_FEATURE_SCALE,
    TabularDataConfig,
    make_tabular_dataloaders,
)

DATASETS = {
    "iris": ("data/tabular/iris.csv", 150, 3),
    "wine": ("data/tabular/wine.csv", 178, 3),
    "seeds": ("data/tabular/seeds.csv", 210, 3),
    "breast_cancer": ("data/tabular/breast_cancer.csv", 569, 2),
}


def _cfg(path, **kw):
    return TabularDataConfig(path=path, target_col="target", batch_size=16, **kw)


def _tensors(loader):
    xs = [b[0].squeeze(1) for b in loader]
    ys = [b[1] for b in loader]
    return torch.cat(xs), torch.cat(ys)


def _rows(x):
    """Hashable rows, for overlap checks."""
    return {tuple(np.round(r, 6)) for r in x.numpy().tolist()}


@pytest.fixture(autouse=True)
def _reset_warn():
    tdm._EXAQC_WARNED = False


# ---------------------------------------------------------------- clean mode

@pytest.mark.parametrize("name", sorted(DATASETS))
def test_clean_split_sizes_are_70_15_15(name):
    require_tabular(name)
    path, n, _ = DATASETS[name]
    _, _, _, meta = make_tabular_dataloaders(
        _cfg(path, train_ratio=0.7, val_ratio=0.15, split_mode="clean"))

    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == n
    assert meta["n_train"] / n == pytest.approx(0.70, abs=0.02)
    assert meta["n_val"] / n == pytest.approx(0.15, abs=0.02)
    assert meta["n_test"] / n == pytest.approx(0.15, abs=0.02)
    assert meta["split_mode"] == "clean"
    assert meta["val_is_test"] is False


def _source_duplicate_rows(path):
    """
    Feature rows that appear more than once in the CSV itself.

    Splits are compared by value, not by index, so a dataset that genuinely
    contains identical rows (iris has one such pair) would look like an overlap.
    Those rows are excluded; anything else overlapping is a real leak.
    """
    import pandas as pd
    df = pd.read_csv(path)
    feat = [c for c in df.columns if c != "target"]
    dup = df[df.duplicated(subset=feat, keep=False)]
    return {tuple(np.round(r, 6)) for r in dup[feat].to_numpy(dtype=np.float32).tolist()}


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_clean_splits_do_not_overlap(name):
    require_tabular(name)
    path, n, _ = DATASETS[name]
    tr, va, te, _ = make_tabular_dataloaders(
        _cfg(path, train_ratio=0.7, val_ratio=0.15, split_mode="clean", normalize=False))

    rtr, rva, rte = (_rows(_tensors(l)[0]) for l in (tr, va, te))
    allowed = _source_duplicate_rows(path)

    assert not ((rtr & rte) - allowed), "train and test share samples"
    assert not ((rtr & rva) - allowed), "train and val share samples"
    assert not ((rva & rte) - allowed), "val and test share samples"
    # And the splits partition the dataset: every sample lands somewhere, once.
    assert len(_tensors(tr)[1]) + len(_tensors(va)[1]) + len(_tensors(te)[1]) == n


def _dataset_proportions(path, k):
    import pandas as pd
    counts = pd.read_csv(path)["target"].value_counts(normalize=True).sort_index()
    return counts.tolist()


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_clean_split_is_stratified(name):
    """
    Each split must mirror the WHOLE dataset's class proportions -- not be
    balanced. Breast cancer is 37/63, so a balance assertion would be wrong.
    """
    require_tabular(name)
    path, n, k = DATASETS[name]
    tr, va, te, meta = make_tabular_dataloaders(
        _cfg(path, train_ratio=0.7, val_ratio=0.15, split_mode="clean"))
    overall = _dataset_proportions(path, k)

    for loader in (tr, va, te):
        _, y = _tensors(loader)
        present = torch.bincount(y, minlength=k)
        assert (present > 0).all(), f"a class is missing from a split: {present}"
        props = (present.float() / len(y)).tolist()
        for got, want in zip(props, overall):
            assert got == pytest.approx(want, abs=0.05), \
                f"{name}: split proportions {props} stray from dataset {overall}"


def test_clean_scaler_is_fitted_on_train_only():
    require_tabular("iris")
    """Train is standardised to ~zero mean; val/test are not, since they were
    transformed with train's statistics."""
    tr, va, _, _ = make_tabular_dataloaders(
        _cfg(DATASETS["iris"][0], split_mode="clean", normalize=True))
    xtr, _ = _tensors(tr)
    assert abs(float(xtr.mean())) < 1e-4
    xva, _ = _tensors(va)
    assert float(xva.mean()) != pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- exaqc mode

@pytest.mark.parametrize("name", sorted(DATASETS))
def test_exaqc_split_is_80_20(name):
    require_tabular(name)
    path, n, _ = DATASETS[name]
    _, _, _, meta = make_tabular_dataloaders(_cfg(path, split_mode="exaqc"))

    assert meta["n_train"] + meta["n_test"] == n
    assert meta["n_train"] / n == pytest.approx(0.80, abs=0.02)
    assert meta["n_test"] / n == pytest.approx(0.20, abs=0.02)
    assert meta["split_mode"] == "exaqc"
    assert meta["val_is_test"] is True


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_exaqc_val_and_test_are_identical(name):
    require_tabular(name)
    path, _, _ = DATASETS[name]
    _, va, te, meta = make_tabular_dataloaders(_cfg(path, split_mode="exaqc"))

    xva, yva = _tensors(va)
    xte, yte = _tensors(te)
    assert torch.equal(xva, xte), "val and test differ; EXAQC uses one holdout"
    assert torch.equal(yva, yte)
    assert meta["n_val"] == meta["n_test"]


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_exaqc_features_are_scaled_to_zero_pi(name):
    require_tabular(name)
    path, _, _ = DATASETS[name]
    tr, _, te, meta = make_tabular_dataloaders(_cfg(path, split_mode="exaqc"))

    xtr, _ = _tensors(tr)
    xte, _ = _tensors(te)
    both = torch.cat([xtr, xte])
    assert float(both.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(both.max()) == pytest.approx(EXAQC_FEATURE_SCALE, abs=1e-5)
    # Fitted on the full dataset: every feature attains both ends somewhere.
    assert float(both.min(dim=0).values.max()) == pytest.approx(0.0, abs=1e-5)
    assert float(both.max(dim=0).values.min()) == pytest.approx(math.pi, abs=1e-5)
    assert meta["scaling"] == "minmax_x_pi_full_dataset"


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_exaqc_split_is_stratified(name):
    require_tabular(name)
    path, _, k = DATASETS[name]
    tr, _, te, _ = make_tabular_dataloaders(_cfg(path, split_mode="exaqc"))
    overall = _dataset_proportions(path, k)
    for loader in (tr, te):
        _, y = _tensors(loader)
        present = torch.bincount(y, minlength=k)
        assert (present > 0).all()
        for got, want in zip((present.float() / len(y)).tolist(), overall):
            assert got == pytest.approx(want, abs=0.05)


def test_exaqc_warns_about_val_being_test(caplog):
    require_tabular("iris")
    seen = []
    from loguru import logger
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        make_tabular_dataloaders(_cfg(DATASETS["iris"][0], split_mode="exaqc"))
    finally:
        logger.remove(sink)

    assert seen, "exaqc mode did not warn"
    text = seen[0]
    assert "validation and test" in text
    assert "not a held-out estimate" in text.lower() or "NOT a held-out" in text


# ---------------------------------------------------------------- misc

def test_unknown_split_mode_raises():
    require_tabular("iris")
    with pytest.raises(ValueError, match="split_mode"):
        make_tabular_dataloaders(_cfg(DATASETS["iris"][0], split_mode="holdout"))


def test_default_split_mode_is_clean():
    # The dataclass default needs no data; the loader half does.
    assert TabularDataConfig(path="x").split_mode == "clean"
    require_tabular("iris")
    _, _, _, meta = make_tabular_dataloaders(_cfg(DATASETS["iris"][0]))
    assert meta["split_mode"] == "clean"


def test_same_seed_gives_same_split():
    require_tabular("iris")
    a = make_tabular_dataloaders(_cfg(DATASETS["iris"][0], random_seed=3))[2]
    b = make_tabular_dataloaders(_cfg(DATASETS["iris"][0], random_seed=3))[2]
    c = make_tabular_dataloaders(_cfg(DATASETS["iris"][0], random_seed=4))[2]
    assert torch.equal(_tensors(a)[0], _tensors(b)[0])
    assert not torch.equal(_tensors(a)[0], _tensors(c)[0])
