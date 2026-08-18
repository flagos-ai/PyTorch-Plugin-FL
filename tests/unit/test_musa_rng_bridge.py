"""Unit coverage for the MUSA FlagGems RNG reservation bridge."""

import sys
import types

import torch_fl


def test_flaggems_philox_uses_flagos_reservations(monkeypatch):
    calls = []

    def reserve_seed(device, generator=None):
        calls.append((device, generator))
        return (1 << 63) + len(calls)

    random_utils = types.ModuleType("flag_gems.utils.random_utils")
    random_utils.torch_device_fn = types.SimpleNamespace(current_device=lambda: 3)
    random_utils.philox_backend_seed_offset = lambda increment, generator=None: (
        999,
        increment,
    )
    utils = types.ModuleType("flag_gems.utils")
    utils.random_utils = random_utils
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.__path__ = []
    flag_gems.utils = utils

    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.utils", utils)
    monkeypatch.setitem(sys.modules, "flag_gems.utils.random_utils", random_utils)
    monkeypatch.setattr(torch_fl, "_build_accelerator", lambda: "musa")
    monkeypatch.setenv("FLAGOS_USE_FLAGGEMS", "1")
    monkeypatch.setattr(torch_fl.flagos._C, "_reserve_rng_seed", reserve_seed)

    torch_fl._patch_flaggems_philox()

    assert random_utils.philox_backend_seed_offset(128) == (-(1 << 63) + 1, 0)
    generator = types.SimpleNamespace(
        device=types.SimpleNamespace(type="flagos", index=2)
    )
    assert random_utils.philox_backend_seed_offset(256, generator) == (
        -(1 << 63) + 2,
        0,
    )
    assert calls == [(3, None), (2, generator)]


def test_flaggems_philox_preserves_non_flagos_generators(monkeypatch):
    sentinel = object()
    random_utils = types.ModuleType("flag_gems.utils.random_utils")
    random_utils.torch_device_fn = types.SimpleNamespace(current_device=lambda: 0)
    random_utils.philox_backend_seed_offset = lambda increment, generator=None: (
        sentinel,
        increment,
    )
    utils = types.ModuleType("flag_gems.utils")
    utils.random_utils = random_utils
    flag_gems = types.ModuleType("flag_gems")
    flag_gems.__path__ = []
    flag_gems.utils = utils

    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setitem(sys.modules, "flag_gems.utils", utils)
    monkeypatch.setitem(sys.modules, "flag_gems.utils.random_utils", random_utils)
    monkeypatch.setattr(torch_fl, "_build_accelerator", lambda: "musa")
    monkeypatch.setenv("FLAGOS_USE_FLAGGEMS", "1")

    torch_fl._patch_flaggems_philox()

    generator = types.SimpleNamespace(
        device=types.SimpleNamespace(type="cpu", index=None)
    )
    assert random_utils.philox_backend_seed_offset(17, generator) == (sentinel, 17)
