"""
auto_detect_device must keep returning exactly what it returned before; the only
addition is a one-time warning when it lands on 'mps'.

eval.device: auto resolves to 'mps' on Apple Silicon, and under torch 2.0.1 the
spawned ProcessBackend worker then dies with BrokenProcessPool and no Python
traceback. The resolution is deliberately unchanged -- other call sites depend on
it -- so the warning is the only signal, and configs pin cpu themselves.
"""

import pytest

from nas_ts.utils import devices


@pytest.fixture(autouse=True)
def _reset_warn_flag(monkeypatch):
    monkeypatch.setattr(devices, "_MPS_WARNED", False)


def _fake_backends(monkeypatch, cuda=False, mps=False):
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(devices.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(devices.torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("preferred,cuda,mps,expected", [
    ("auto", False, False, "cpu"),
    ("auto", False, True, "mps"),
    ("auto", True, False, "cuda:0"),
    (None, False, True, "mps"),
    ("cpu", False, True, "cpu"),      # explicit cpu wins over available mps
    ("mps", False, True, "mps"),
    ("cuda", False, False, "cpu"),    # cuda asked for, none present
    ("cuda:1", False, False, "cuda:1"),
])
def test_resolution_is_unchanged(monkeypatch, preferred, cuda, mps, expected):
    _fake_backends(monkeypatch, cuda=cuda, mps=mps)
    assert devices.auto_detect_device(preferred) == expected


def test_mps_warns_once_not_per_call(monkeypatch):
    _fake_backends(monkeypatch, cuda=False, mps=True)
    seen = []
    monkeypatch.setattr(devices, "logger",
                        type("L", (), {"warning": staticmethod(lambda m: seen.append(m))})())

    for _ in range(5):
        assert devices.auto_detect_device("auto") == "mps"

    assert len(seen) == 1, f"expected one warning, got {len(seen)}"
    assert "mps" in seen[0].lower()
    assert "eval.device=cpu" in seen[0]


def test_cpu_resolution_does_not_warn(monkeypatch):
    _fake_backends(monkeypatch, cuda=False, mps=False)
    seen = []
    monkeypatch.setattr(devices, "logger",
                        type("L", (), {"warning": staticmethod(lambda m: seen.append(m))})())

    assert devices.auto_detect_device("auto") == "cpu"
    assert seen == []


def test_iris_config_pins_cpu():
    import yaml
    cfg = yaml.safe_load(open("configs/iris_test.yml"))
    assert cfg["eval"]["device"] == "cpu"
