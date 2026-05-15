"""
MACA compatibility layer for torch.cuda on MetaX (Muxi) hardware.

On MetaX systems, PyTorch's bundled libcudart.so.12 (CUDA 12.x) is ABI-incompatible
with MACA's cu-bridge (CUDA 11.6 compat). Basic CUDA calls (device_count, is_available)
work via LD_PRELOAD of libsymbol_cu.so, but get_device_properties fails because
PyTorch calls cudaGetDeviceProperties_v2 (CUDA 12 API) which MACA doesn't provide.

This module monkey-patches torch.cuda.get_device_properties and get_device_name
to use MACA's native mcruntime API via ctypes, bypassing the incompatible C++ path.

Usage:
    Before importing flag_gems or any code that calls torch.cuda.get_device_properties:

        from torch_fl._maca_compat import patch_torch_cuda_for_maca
        patch_torch_cuda_for_maca()

Environment:
    Requires LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so and MACA SDK installed.
    Set MACA_PATH env var if MACA is not at /opt/maca.
"""

import ctypes
import os
import warnings
from dataclasses import dataclass
from typing import Union

import torch


def _find_maca_path():
    """Find the MACA SDK root path."""
    for env_var in ("MACA_PATH", "MACA_HOME"):
        path = os.environ.get(env_var)
        if path and os.path.isdir(path):
            return path
    for default in ("/opt/maca", "/opt/maca-3.3.0"):
        if os.path.isdir(default):
            return default
    return None


def _load_mcruntime(maca_path):
    """Load MACA mcruntime library via ctypes."""
    lib_path = os.path.join(maca_path, "lib", "libmcruntime.so")
    if not os.path.isfile(lib_path):
        return None
    try:
        return ctypes.CDLL(lib_path)
    except OSError:
        return None


@dataclass
class _MacaDeviceProperties:
    """Minimal device properties matching torch.cuda._CudaDeviceProperties interface."""

    name: str = ""
    major: int = 0
    minor: int = 0
    total_memory: int = 0
    multi_processor_count: int = 0
    is_integrated: bool = False
    is_multi_gpu_board: bool = False
    warp_size: int = 64
    max_threads_per_multi_processor: int = 0
    regs_per_multiprocessor: int = 0
    shared_memory_per_block: int = 0
    shared_memory_per_block_optin: int = 0
    shared_memory_per_multiprocessor: int = 0
    L2_cache_size: int = 0
    pci_bus_id: str = ""
    pci_device_id: str = ""
    pci_domain_id: str = ""
    uuid: str = ""
    gcnArchName: str = ""


def _query_maca_device_properties(mcruntime, device_index):
    """Query device properties from MACA native API."""
    props = _MacaDeviceProperties()

    # mcDeviceGetName(char *name, int len, int device)
    name_buf = ctypes.create_string_buffer(256)
    ret = mcruntime.mcDeviceGetName(name_buf, 256, device_index)
    if ret == 0:
        props.name = name_buf.value.decode("utf-8", errors="replace")

    # mcDeviceGetAttribute(int *value, int attr, int device)
    mcDeviceGetAttribute = mcruntime.mcDeviceGetAttribute
    mcDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    mcDeviceGetAttribute.restype = ctypes.c_int

    val = ctypes.c_int(0)

    def get_attr(attr_id):
        ret = mcDeviceGetAttribute(ctypes.byref(val), attr_id, device_index)
        return val.value if ret == 0 else 0

    props.multi_processor_count = get_attr(16)
    props.warp_size = get_attr(10)
    props.max_threads_per_multi_processor = get_attr(39)
    props.shared_memory_per_block = get_attr(8)
    props.shared_memory_per_multiprocessor = get_attr(81)

    # Major/minor from MACA may not map to CUDA compute capability.
    # Use reasonable defaults for MetaX GPUs.
    raw_major = get_attr(21)
    raw_minor = get_attr(22)
    if raw_major > 100:
        # MACA returns non-standard values; use a sensible default
        props.major = 8
        props.minor = 0
    else:
        props.major = raw_major
        props.minor = raw_minor

    # Get total memory
    mcruntime.mcSetDevice(device_index)
    free_mem = ctypes.c_size_t(0)
    total_mem = ctypes.c_size_t(0)
    mcMemGetInfo = mcruntime.mcMemGetInfo
    mcMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    mcMemGetInfo.restype = ctypes.c_int
    ret = mcMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem))
    if ret == 0:
        props.total_memory = total_mem.value // (1024 * 1024)  # in MiB

    return props


# Cache for device properties
_maca_props_cache = {}
_mcruntime = None
_patched = False


def is_maca_available():
    """Check if MACA runtime is available."""
    maca_path = _find_maca_path()
    if maca_path is None:
        return False
    return _load_mcruntime(maca_path) is not None


def patch_torch_cuda_for_maca():
    """
    Monkey-patch torch.cuda functions to work on MACA hardware.

    This patches:
    - torch.cuda.get_device_properties: uses MACA native API
    - torch.cuda.get_device_name: uses MACA native API
    - torch.cuda._lazy_init: skips capability check
    - torch.cuda.get_device_capability: returns sensible defaults

    Must be called before importing flag_gems or other code that uses
    torch.cuda.get_device_properties.
    """
    global _mcruntime, _patched

    if _patched:
        return True

    maca_path = _find_maca_path()
    if maca_path is None:
        warnings.warn("MACA SDK not found, skipping torch.cuda patches")
        return False

    _mcruntime = _load_mcruntime(maca_path)
    if _mcruntime is None:
        warnings.warn(f"Cannot load libmcruntime.so from {maca_path}/lib/")
        return False

    # Skip PyTorch's CUDA capability check (it fails because MACA reports
    # CUDA 11.6 but PyTorch expects CUDA 12.0+ runtime)
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()

    def _patched_get_device_properties(
        device: Union[torch.device, int, str, None] = None,
    ):
        """Get device properties from MACA native API."""
        if device is None:
            device_index = torch.cuda.current_device()
        elif isinstance(device, torch.device):
            device_index = device.index if device.index is not None else 0
        elif isinstance(device, str):
            device_index = int(device.split(":")[-1]) if ":" in device else 0
        else:
            device_index = int(device)

        if device_index not in _maca_props_cache:
            _maca_props_cache[device_index] = _query_maca_device_properties(
                _mcruntime, device_index
            )
        return _maca_props_cache[device_index]

    def _patched_get_device_name(
        device: Union[torch.device, int, str, None] = None,
    ) -> str:
        """Get device name from MACA native API."""
        props = _patched_get_device_properties(device)
        return props.name

    def _patched_get_device_capability(
        device: Union[torch.device, int, str, None] = None,
    ):
        """Get device capability from MACA (returns sensible defaults)."""
        props = _patched_get_device_properties(device)
        return (props.major, props.minor)

    torch.cuda.get_device_properties = _patched_get_device_properties
    torch.cuda.get_device_name = _patched_get_device_name
    torch.cuda.get_device_capability = _patched_get_device_capability

    # Patch basic CUDA tensor operations that use NVIDIA-specific kernels.
    # On MACA, PyTorch's built-in CUDA kernels (fill_, zero_, copy_) fail
    # because they are compiled for NVIDIA GPUs. We replace them with
    # implementations using MACA's cu-bridge runtime API (cudaMemset, cudaMemcpy).
    _patch_cuda_tensor_ops(maca_path)

    _patched = True
    return True


_cudaMemcpyHostToDevice = 1

# Cached cudart handle
_cudart = None


def _get_cudart():
    """Get the MACA-compatible CUDA runtime library (cached)."""
    global _cudart
    if _cudart is not None:
        return _cudart
    # libsymbol_cu.so provides cuda* API functions compatible with MACA
    try:
        _cudart = ctypes.CDLL("libsymbol_cu.so")
    except OSError:
        maca_path = _find_maca_path()
        if maca_path:
            _cudart = ctypes.CDLL(os.path.join(maca_path, "lib", "libsymbol_cu.so"))
    return _cudart


def _patch_cuda_tensor_ops(maca_path):
    """
    Patch PyTorch's CUDA tensor ops to use MACA runtime API instead of
    NVIDIA-specific CUDA kernels.

    Patched operations:
    - Tensor.zero_(): uses cudaMemsetAsync (contiguous only)
    - Tensor.fill_(scalar): uses cudaMemsetAsync for 0, cudaMemcpy for others (contiguous only)
    """
    cudart = _get_cudart()
    if cudart is None:
        return

    # Setup cudaMemsetAsync
    cudaMemsetAsync = cudart.cudaMemsetAsync
    cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cudaMemsetAsync.restype = ctypes.c_int

    # Setup cudaMemcpyAsync
    cudaMemcpyAsync = cudart.cudaMemcpyAsync
    cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    cudaMemcpyAsync.restype = ctypes.c_int

    # Setup cudaStreamSynchronize
    cudaStreamSynchronize = cudart.cudaStreamSynchronize
    cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudaStreamSynchronize.restype = ctypes.c_int

    def _maca_copy_to_cuda(dst, src):
        """Copy src (CPU, contiguous) into dst (CUDA, contiguous) via cudaMemcpyAsync."""
        nbytes = src.nelement() * src.element_size()
        ret = cudaMemcpyAsync(
            dst.data_ptr(), src.data_ptr(), nbytes, _cudaMemcpyHostToDevice, None
        )
        if ret != 0:
            warnings.warn(f"cudaMemcpyAsync failed with error {ret}")
        cudaStreamSynchronize(None)

    _orig_zero = torch.Tensor.zero_

    def _maca_zero_(self):
        """zero_() using cudaMemsetAsync for CUDA tensors on MACA."""
        if not self.is_cuda:
            return _orig_zero(self)
        if not self.is_contiguous():
            raise NotImplementedError(
                "MACA compat: zero_() on non-contiguous CUDA tensors is not supported. "
                "Call .contiguous() first."
            )
        ptr = self.data_ptr()
        nbytes = self.nelement() * self.element_size()
        ret = cudaMemsetAsync(ptr, 0, nbytes, None)
        if ret != 0:
            cpu_zeros = torch.zeros(self.shape, dtype=self.dtype, device="cpu")
            _maca_copy_to_cuda(self, cpu_zeros)
        return self

    torch.Tensor.zero_ = _maca_zero_

    _orig_fill = torch.Tensor.fill_

    def _maca_fill_(self, value):
        """fill_() for CUDA tensors on MACA."""
        if not self.is_cuda:
            return _orig_fill(self, value)
        if value == 0 and self.is_contiguous():
            return _maca_zero_(self)
        # Create on CPU and copy to CUDA
        cpu_t = torch.full(self.shape, value, dtype=self.dtype, device="cpu")
        if not self.is_contiguous():
            raise NotImplementedError(
                "MACA compat: fill_() on non-contiguous CUDA tensors is not supported. "
                "Call .contiguous() first."
            )
        _maca_copy_to_cuda(self, cpu_t)
        return self

    torch.Tensor.fill_ = _maca_fill_
