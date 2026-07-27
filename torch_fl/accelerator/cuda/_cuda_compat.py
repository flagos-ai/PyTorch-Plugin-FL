# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic NVIDIA CUDA compatibility layer for torch.cuda under the flagos backend.

Under the external-libtorch scheme the pip torch is CPU-only
(``torch.__version__`` ends with ``+cpu``): its ``torch.cuda`` Python bindings
were compiled WITHOUT CUDA, so ``torch.cuda.is_available()`` is ``False`` and
``torch.cuda._lazy_init()`` raises "Torch not compiled with CUDA enabled". That
cannot be fixed by ``LD_PRELOAD``-ing ``libtorch_cuda.so`` -- the Python layer
was frozen at compile time.

Triton, however, does NOT use torch's CUDA Python layer to compile/launch
kernels: it uses its own C extension plus the system ``libcuda.so`` (the NVIDIA
driver). So FlagGems' Triton kernels can run correctly as long as we make
torch.cuda's *probe* functions report a real device. This module monkey-patches
those probes, sourcing real values from the CUDA Driver API (``libcuda.so``,
always present alongside an NVIDIA driver) via ctypes -- no CUDA runtime, no
torch CUDA build required.

Enabled by default from ``torch_fl.__init__`` when a generic NVIDIA GPU is
detected (not MetaX, not Ascend). Disable with ``FLAGOS_DISABLE_CUDA_SHIM=1``.

Modeled on ``torch_fl/accelerator/metax/_metax_compat.py``.
"""

import ctypes
import os
import warnings
from dataclasses import dataclass, field
from typing import Union

import torch


_patched = False
_cuda = None  # cached libcuda.so handle
_cudart = None  # cached libcudart.so handle (for synchronize)
_props_cache = {}

# device_index -> torch.Generator(device="cuda"), one per device. See
# _get_cuda_generator / _CudaDefaultGenerators below.
_cuda_generators = {}


def _get_cuda_generator(idx):
    """Lazily build one CUDA generator per device (the flaggems RNG source).

    flag_gems' ``philox_backend_seed_offset`` (nvidia ``device_name="cuda"``)
    reads ``torch.cuda.default_generators[device]`` and unpacks its 16-byte
    state as 2x int64 ``(seed, offset)`` -- the CUDA generator's philox layout.
    We install these as ``torch.cuda.default_generators`` so gems' generator-less
    RNG ops find a real, seedable generator instead of the empty tuple the
    CPU-torch wheel ships (which used to force the philox monkeypatch).

    Lazy because at import time the external ``libtorch_cuda.so`` is not yet
    wired into ATen -- ``torch.Generator(device="cuda")`` raises "Cannot get
    CUDA generator without ATen_cuda library". By the time any RNG op runs, cuda
    is live and construction succeeds. Seeded from ``torch.initial_seed()`` so a
    ``torch.manual_seed(...)`` issued before first use is honoured.
    """
    gen = _cuda_generators.get(idx)
    if gen is None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(torch.initial_seed())
        _cuda_generators[idx] = gen
    return gen


class _CudaDefaultGenerators:
    """list-like stand-in for ``torch.cuda.default_generators``.

    Indexing yields a real (lazily created) per-device CUDA generator; ``len``
    reports the device count so flag_gems' ``len(default_generators) == 0``
    guard is False and it uses the generator instead of erroring.
    """

    def __getitem__(self, idx):
        return _get_cuda_generator(int(idx))

    def __len__(self):
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        return max(n, 1)


# ---- CUDA Driver API (libcuda.so) constants ----
# CUdevice_attribute enum values (cuda.h). Confirmed against A100 (sm_80).
_ATTR_CC_MAJOR = 75
_ATTR_CC_MINOR = 76
_ATTR_MP_COUNT = 16
_ATTR_L2_CACHE_SIZE = 38
_ATTR_MAX_THREADS_PER_MP = 39
_ATTR_WARP_SIZE = 10


@dataclass
class _CudaDeviceProperties:
    """Minimal stand-in for torch.cuda._CudaDeviceProperties.

    Exposes the fields FlagGems reads: ``name``, ``major``, ``minor``,
    ``multi_processor_count``, ``L2_cache_size``, ``total_memory``,
    ``warp_size``, ``max_threads_per_multi_processor``.
    """

    name: str = ""
    major: int = 8
    minor: int = 0
    total_memory: int = 0
    multi_processor_count: int = 108
    L2_cache_size: int = 40 * 1024 * 1024
    warp_size: int = 32
    max_threads_per_multi_processor: int = 2048
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    gcnArchName: str = ""

    def __repr__(self):
        return (
            f"_CudaDeviceProperties(name='{self.name}', "
            f"major={self.major}, minor={self.minor}, "
            f"total_memory={self.total_memory // (1024 * 1024)}MB, "
            f"multi_processor_count={self.multi_processor_count})"
        )


def _load_libcuda():
    """Load the NVIDIA driver library (libcuda.so) via ctypes, cached."""
    global _cuda
    if _cuda is not None:
        return _cuda
    for name in ("libcuda.so", "libcuda.so.1"):
        try:
            _cuda = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _cuda is not None:
        # cuInit(0) is idempotent and required before other Driver API calls.
        try:
            _cuda.cuInit(0)
        except Exception:
            _cuda = None
    return _cuda


def is_nvidia_cuda_available() -> bool:
    """True if a generic NVIDIA GPU is reachable via the driver API."""
    cuda = _load_libcuda()
    if cuda is None:
        return False
    count = ctypes.c_int(0)
    try:
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return False
    except Exception:
        return False
    return count.value > 0


def _device_index(device: Union[torch.device, int, str, None]) -> int:
    if device is None:
        return 0
    if isinstance(device, torch.device):
        return device.index if device.index is not None else 0
    if isinstance(device, str):
        return int(device.split(":")[-1]) if ":" in device else 0
    return int(device)


def _query_device_properties(device_index: int) -> _CudaDeviceProperties:
    """Query device properties from the CUDA Driver API."""
    props = _CudaDeviceProperties()
    cuda = _load_libcuda()
    if cuda is None:
        return props

    dev = ctypes.c_int(0)
    if cuda.cuDeviceGet(ctypes.byref(dev), device_index) != 0:
        return props

    # Name
    name_buf = ctypes.create_string_buffer(256)
    if cuda.cuDeviceGetName(name_buf, 256, dev) == 0:
        props.name = name_buf.value.decode("utf-8", errors="replace")

    val = ctypes.c_int(0)

    def attr(attr_id, default):
        if cuda.cuDeviceGetAttribute(ctypes.byref(val), attr_id, dev) == 0:
            return val.value
        return default

    props.major = attr(_ATTR_CC_MAJOR, props.major)
    props.minor = attr(_ATTR_CC_MINOR, props.minor)
    props.multi_processor_count = attr(_ATTR_MP_COUNT, props.multi_processor_count)
    props.L2_cache_size = attr(_ATTR_L2_CACHE_SIZE, props.L2_cache_size)
    props.warp_size = attr(_ATTR_WARP_SIZE, props.warp_size)
    props.max_threads_per_multi_processor = attr(
        _ATTR_MAX_THREADS_PER_MP, props.max_threads_per_multi_processor
    )

    total = ctypes.c_size_t(0)
    try:
        if cuda.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0:
            props.total_memory = total.value
    except Exception:
        pass

    return props


def _get_props(device=None) -> _CudaDeviceProperties:
    idx = _device_index(device)
    if idx not in _props_cache:
        _props_cache[idx] = _query_device_properties(idx)
    return _props_cache[idx]


def _load_cudart():
    """Load libcudart.so for cudaDeviceSynchronize, cached."""
    global _cudart
    if _cudart is not None:
        return _cudart
    try:
        _cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        try:
            _cudart = ctypes.CDLL(f"{cuda_home}/lib64/libcudart.so")
        except OSError:
            _cudart = None
    return _cudart


class _StreamShim:
    """Minimal stream object exposing ``.cuda_stream`` for triton.

    Uses the null/default stream (0), consistent with the boxing path where the
    caching allocator is given ``stream=nullptr``.
    """

    def __init__(self, index=0):
        self.cuda_stream = 0
        self.device_index = index

    def synchronize(self):
        _synchronize()


def _synchronize(device=None):
    cudart = _load_cudart()
    if cudart is not None:
        try:
            cudart.cudaDeviceSynchronize()
            return
        except Exception:
            pass
    # Fall back to driver API context sync.
    cuda = _load_libcuda()
    if cuda is not None:
        try:
            cuda.cuCtxSynchronize()
        except Exception:
            pass


def patch_torch_cuda_for_flagos():
    """Monkey-patch torch.cuda probes to report a real NVIDIA GPU.

    Must be called before importing flag_gems (which reads
    ``torch.cuda.get_device_name()`` at import).
    """
    global _patched
    if _patched:
        return True

    if not is_nvidia_cuda_available():
        warnings.warn(
            "torch_fl: no NVIDIA GPU reachable via libcuda.so; "
            "skipping torch.cuda shim"
        )
        return False

    _flagos = torch.flagos if hasattr(torch, "flagos") else None

    def _device_count():
        cuda = _load_libcuda()
        if cuda is None:
            return 0
        count = ctypes.c_int(0)
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return 0
        return count.value

    def _current_device():
        # Route through the flagos runtime so the notion of "current device"
        # stays consistent with the PrivateUse1 backend.
        if _flagos is not None:
            try:
                return _flagos.current_device()
            except Exception:
                pass
        return 0

    def _set_device(device):
        idx = _device_index(device)
        if _flagos is not None:
            try:
                _flagos.set_device(idx)
            except Exception:
                pass

    # --- probes ---
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = _device_count
    torch.cuda.current_device = _current_device
    torch.cuda.set_device = _set_device
    torch.cuda.get_device_properties = _get_props
    torch.cuda.get_device_name = lambda device=None: _get_props(device).name
    torch.cuda.get_device_capability = lambda device=None: (
        _get_props(device).major,
        _get_props(device).minor,
    )
    torch.cuda.synchronize = _synchronize

    # _lazy_init must be a no-op; the real one raises on CPU torch.
    torch.cuda._lazy_init = lambda: None
    if hasattr(torch.cuda, "_initialized"):
        torch.cuda._initialized = True
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    # Device context: extract index for flagos/privateuseone; forward to driver.
    _orig_device_init = torch.cuda.device.__init__

    def _patched_device_init(self, device):
        if hasattr(device, "type") and hasattr(device, "index"):
            if device.type in ("privateuseone", "flagos"):
                device = device.index if device.index is not None else 0
        try:
            return _orig_device_init(self, device)
        except Exception:
            # CPU torch's device ctx may reject; store index for our exchange.
            self.idx = _device_index(device)
            self.prev_idx = -1

    torch.cuda.device.__init__ = _patched_device_init

    def _exchange_device(idx):
        if idx < 0:
            return -1
        prev = _current_device()
        _set_device(idx)
        return prev

    torch.cuda._exchange_device = _exchange_device
    torch.cuda._maybe_exchange_device = _exchange_device

    # Streams for triton raw-stream lookup.
    torch.cuda.current_stream = lambda device=None: _StreamShim(_device_index(device))
    torch.cuda.default_stream = lambda device=None: _StreamShim(_device_index(device))

    # triton reads torch._C._cuda_getCurrentRawStream(idx) -> raw handle.
    try:
        torch._C._cuda_getCurrentRawStream = lambda idx=0: 0
    except Exception:
        pass
    try:
        torch._C._cuda_synchronize = lambda: _synchronize()
    except Exception:
        pass

    # Seeding / RNG source: the CPU-torch wheel ships an EMPTY
    # torch.cuda.default_generators, and torch.manual_seed() -> [nothing on
    # cuda], so flag_gems' generator-less RNG ops (rand/randn/uniform_/...) had
    # no seedable source and were not reproducible. Install per-device CUDA
    # generators (philox 2x int64 state, exactly what gems'
    # philox_backend_seed_offset unpacks) and route cuda seeding to them, so
    # torch.manual_seed(s) -> torch.cuda.manual_seed_all(s) reseeds them and
    # gems RNG becomes reproducible. This is the SAME shape the metax branch
    # uses; it replaces the old _patch_flaggems_philox monkeypatch.
    def _manual_seed(seed):
        seed = int(seed)
        idx = _current_device()
        try:
            _get_cuda_generator(idx).manual_seed(seed)
        except Exception:
            pass

    def _manual_seed_all(seed):
        seed = int(seed)
        try:
            for i in range(max(_device_count(), 1)):
                _get_cuda_generator(i).manual_seed(seed)
        except Exception:
            pass

    torch.cuda.manual_seed = _manual_seed
    torch.cuda.manual_seed_all = _manual_seed_all
    try:
        torch.cuda.default_generators = _CudaDefaultGenerators()
    except Exception:
        pass

    _patch_triton_do_bench()

    _patched = True
    return True


def _patch_triton_do_bench():
    """Replace triton.testing.do_bench to avoid CUDA Event timing.

    triton's autotuner benchmarks kernels with ``torch.cuda.Event(
    enable_timing=True)`` and ``torch.empty(device='cuda')``, both of which fail
    on CPU torch. We time with a wall clock instead. Timing only affects
    autotune config *selection*, not kernel correctness -- kernels still run on
    the real GPU via the system libcuda.so.
    """
    try:
        import triton
        import triton.testing
    except ImportError:
        return

    import time
    import statistics

    def _do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None,
                  return_mode="mean", **kwargs):
        # Warmup
        fn()
        _synchronize()
        # A few timed reps with a wall clock.
        n_rep = 5
        times = []
        for _ in range(n_rep):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            t0 = time.perf_counter()
            fn()
            _synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

        if quantiles is not None:
            times_sorted = sorted(times)

            def _quantile(q):
                pos = q * (len(times_sorted) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(times_sorted) - 1)
                frac = pos - lo
                return times_sorted[lo] * (1 - frac) + times_sorted[hi] * frac

            ret = [_quantile(q) for q in quantiles]
            return ret[0] if len(ret) == 1 else ret

        if return_mode == "min":
            return min(times)
        if return_mode == "max":
            return max(times)
        if return_mode == "median":
            return statistics.median(times)
        if return_mode == "all":
            return times
        return statistics.mean(times)

    triton.testing.do_bench = _do_bench
    # Some triton versions cache the benchmarker on the driver; refresh it.
    try:
        triton.runtime.driver.active.get_benchmarker = lambda: _do_bench
    except Exception:
        pass
