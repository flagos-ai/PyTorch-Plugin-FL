"""Vendor-routing unit tests for ProcessGroupFlagOS.

Pure-logic tests for the GEMS_VENDOR -> (flagcx devName, view, native backend)
routing in torch_fl.comm.process_group. No GPU / no real backend needed: we
import the module directly (it has no import-time torch_fl._C dependency) and
drive _build_inner with fakes to assert the decision table, not the transport.

Run: pytest tests/unit/test_vendor_routing.py
"""

import sys
import types

import pytest

# Import the module WITHOUT importing torch_fl (which preloads cuda). The module
# only needs torch + torch.distributed at import time.
pg = pytest.importorskip("torch_fl.comm.process_group")


# ---------------------------------------------------------------------------
# Profile table
# ---------------------------------------------------------------------------

def test_all_vendors_have_consistent_profiles():
    for name, prof in pg._VENDOR_PROFILES.items():
        assert prof.flagcx_dev, f"{name} missing flagcx_dev"
        # cuda-ABI vendors must expose the zero-copy cuda view + NCCL fallback;
        # non-cuda vendors must NOT claim the cuda view.
        if prof.flagcx_dev == "cuda":
            assert prof.view == "_flagos_to_cuda_view", name
            assert prof.native == "_try_build_nccl", name
        else:
            assert prof.view is None, (
                f"{name} is not a cuda alias but claims view {prof.view!r}")


def test_unknown_vendor_falls_back_to_default_profile():
    with pytest.warns(UserWarning, match="unknown GEMS_VENDOR"):
        prof = pg._get_profile("totally-made-up")
    assert prof is pg._VENDOR_PROFILES[pg._DEFAULT_VENDOR]


def test_known_cuda_vendors():
    for v in ("nvidia", "metax", "iluvatar", "kunlunxin", "du"):
        prof = pg._get_profile(v)
        assert prof.flagcx_dev == "cuda"
        assert prof.view == "_flagos_to_cuda_view"


# ---------------------------------------------------------------------------
# _build_inner routing (with fakes)
# ---------------------------------------------------------------------------

def _make(monkeypatch, vendor, *, flagcx_ok, native_ok, view_present=True):
    """Configure a fake ProcessGroupFlagOS and stub out its build helpers.

    __init__ (and thus the ProcessGroup C++ base ctor) is bypassed via __new__
    so we can drive _build_inner in isolation without a real store/rank/size.
    """
    obj = pg.ProcessGroupFlagOS.__new__(pg.ProcessGroupFlagOS)
    calls = {"flagcx": False, "native": False}

    def fake_flagcx(self, store, rank, ws, timeout):
        calls["flagcx"] = True
        if flagcx_ok:
            self._inner = object()
        return flagcx_ok

    def fake_nccl(self, store, rank, ws, timeout):
        calls["native"] = True
        if native_ok:
            self._inner = object()
        return native_ok

    def fake_hccl(self, store, rank, ws, timeout):
        calls["native"] = True
        if native_ok:
            self._inner = object()
        return native_ok

    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_flagcx", fake_flagcx)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_nccl", fake_nccl)
    monkeypatch.setattr(pg.ProcessGroupFlagOS, "_try_build_hccl", fake_hccl)

    # Fake torch_fl._C so _resolve_view finds (or misses) the view helper.
    # _resolve_view does `import torch_fl._C as _C`; when torch_fl is already
    # imported that resolves via the parent package attribute, so patch both the
    # sys.modules entry and the attribute on the (possibly real) torch_fl pkg.
    sentinel_view = object()  # unique marker so tests can assert identity
    fake_c = types.ModuleType("torch_fl._C")
    if view_present:
        fake_c._flagos_to_cuda_view = sentinel_view
    monkeypatch.setitem(sys.modules, "torch_fl._C", fake_c)
    torch_fl_pkg = sys.modules.get("torch_fl")
    if torch_fl_pkg is not None:
        monkeypatch.setattr(torch_fl_pkg, "_C", fake_c, raising=False)
    monkeypatch.setenv("GEMS_VENDOR", vendor)
    return obj, calls, sentinel_view


def test_flagcx_preferred_over_native(monkeypatch):
    obj, calls, sentinel = _make(monkeypatch, "nvidia", flagcx_ok=True,
                                 native_ok=True)
    view = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and not calls["native"]
    assert view is sentinel  # cuda view resolved from the (faked) torch_fl._C


def test_native_used_when_flagcx_unavailable(monkeypatch):
    obj, calls, _ = _make(monkeypatch, "nvidia", flagcx_ok=False, native_ok=True)
    view = obj._build_inner(None, 0, 1, None)
    assert calls["flagcx"] and calls["native"]
    assert obj._inner is not None


def test_no_backend_raises(monkeypatch):
    obj, _, _ = _make(monkeypatch, "nvidia", flagcx_ok=False, native_ok=False)
    with pytest.raises(RuntimeError, match="no suitable inner backend"):
        obj._build_inner(None, 0, 1, None)


def test_musa_flagcx_only_no_native_fallback(monkeypatch):
    # musa has native=None: if flagcx fails there is nothing to fall back to.
    obj, calls, _ = _make(monkeypatch, "musa", flagcx_ok=False, native_ok=True,
                          view_present=False)
    with pytest.raises(RuntimeError, match="none wired"):
        obj._build_inner(None, 0, 1, None)
    assert not calls["native"]


def test_musa_flagcx_ok_but_no_view_raises(monkeypatch):
    # musa flagcx path succeeds, but no flagos->musa view is implemented yet ->
    # must fail loudly rather than pass a raw flagos tensor to the backend.
    obj, calls, _ = _make(monkeypatch, "musa", flagcx_ok=True, native_ok=False,
                          view_present=False)
    with pytest.raises(NotImplementedError, match="no flagos->device view"):
        obj._build_inner(None, 0, 1, None)


def test_ascend_uses_hccl_native(monkeypatch):
    # ascend flagcx fails -> native hccl succeeds -> but view is None -> raise
    # NotImplementedError (documents that only the FlagCX(cann) path is viable).
    obj, calls, _ = _make(monkeypatch, "ascend", flagcx_ok=False, native_ok=True,
                          view_present=False)
    with pytest.raises(NotImplementedError, match="no flagos->device view"):
        obj._build_inner(None, 0, 1, None)
    assert calls["native"]  # hccl was attempted
