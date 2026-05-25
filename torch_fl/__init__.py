import sys

from torch_fl.accelerator.maca._maca_cudart_shim import ensure_cudart_shim

ensure_cudart_shim()

import torch  # noqa: E402


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


# Apply MACA compatibility patches before importing FlagGems.
# On MetaX (Muxi) hardware, PyTorch's bundled CUDA 12.x runtime is
# ABI-incompatible with MACA's cu-bridge (CUDA 11.6). This patches
# torch.cuda.get_device_properties/get_device_name to use MACA's
# native mcruntime API, allowing FlagGems initialization to succeed.
from torch_fl.accelerator.maca._maca_compat import (  # noqa: E402
    is_maca_available,
    patch_torch_cuda_for_maca,
)

if is_maca_available():
    patch_torch_cuda_for_maca()


import torch_fl._C  # type: ignore[misc]  # noqa: E402, F401
from . import flagos  # noqa: E402


torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", flagos)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)


# Global library instance to keep registrations alive
_flaggems_lib = None
_autograd_lib = None
_registered_ops = []


def _patch_cuda_device_context():
    """
    Monkey-patch torch.cuda.device to handle flagos devices.

    FlagGems internally calls torch_device_fn.device(tensor.device), but when
    tensor.device is 'flagos:0', torch.cuda.device() fails because it expects
    a CUDA device. This patch wraps torch.cuda.device.__init__ to extract just
    the device index from flagos/privateuseone devices.
    """
    _original_cuda_device_init = torch.cuda.device.__init__

    def _patched_cuda_device_init(self, device):
        # Handle flagos/privateuseone devices by extracting just the index
        if hasattr(device, "type") and hasattr(device, "index"):
            if device.type in ("privateuseone", "flagos"):
                device = device.index if device.index is not None else 0
        return _original_cuda_device_init(self, device)

    torch.cuda.device.__init__ = _patched_cuda_device_init


# Patch torch.cuda.device before FlagGems is used
_patch_cuda_device_context()

# Ensure CUDA runtime is initialized so that CUDACachingAllocator is ready.
# Without this, C++ stubs routing ops to the cuda backend (cuBLAS, etc.)
# would hit "Allocator not initialized for device" errors.
if torch.cuda.is_available():
    torch.cuda.init()




# Cache for CUDA runtime library
_cudart_lib = None
_cudaMemcpy = None


def _get_cudaMemcpy():
    """Get cudaMemcpy function from CUDA runtime library (cached)."""
    global _cudart_lib, _cudaMemcpy
    if _cudaMemcpy is not None:
        return _cudaMemcpy

    import ctypes
    import os

    # Try to load CUDA runtime library
    try:
        _cudart_lib = ctypes.CDLL("libcudart.so")
    except OSError:
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _cudart_lib = ctypes.CDLL(f"{cuda_home}/lib64/libcudart.so")

    _cudaMemcpy = _cudart_lib.cudaMemcpy
    _cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    _cudaMemcpy.restype = ctypes.c_int

    return _cudaMemcpy


def _register_flaggems_operators():
    """
    Register FlagGems operators with the PrivateUse1 (flagos) dispatch key.

    Disabled: Python-layer FlagGems registration is not used.
    All ops are dispatched through the C++ stub path instead.
    """
    global _flaggems_lib, _autograd_lib, _registered_ops
    _registered_ops = []
    return 0


def _register_composite_ops():
    """
    Register CompositeExplicitAutograd ops that cause cpu_fallback segfault.

    Some PyTorch ops are CompositeExplicitAutograd (not CompositeImplicitAutograd),
    meaning they don't auto-decompose for PrivateUse1. They fall through to
    cpu_fallback which segfaults when handling privateuseone tensors.

    We register these manually by implementing them in terms of ops that are
    already registered (like slice_scatter).
    """
    lib = torch.library.Library("aten", "IMPL")

    # slice_backward: used by autograd for tensor slicing (x[..., :n])
    # Implementation mirrors PyTorch's native slice_backward which calls slice_scatter
    def slice_backward_impl(grad_output, input_sizes, dim, start, end, step):
        # Convert SymInt to int for compatibility
        input_sizes = [int(s) for s in input_sizes]
        dim = int(dim)
        start = int(start)
        end = int(end)
        step = int(step)
        # Clamp end to input_sizes[dim] (PyTorch passes large values like sys.maxsize)
        if end > input_sizes[dim]:
            end = input_sizes[dim]
        grad_input = torch.zeros(
            input_sizes, dtype=grad_output.dtype, device=grad_output.device
        )
        return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)

    # lib.impl("slice_backward", slice_backward_impl, "PrivateUse1")

    # log_softmax: decompose into softmax + log to avoid FlagGems Triton kernel
    # that exceeds MACA's 4KB/thread private memory on large vocab dimensions.
    # The softmax kernel already has proper tiling for large N.
    def log_softmax_impl(self, dim, half_to_float=False):
        dtype = torch.float32 if half_to_float else self.dtype
        out = torch.softmax(self.to(torch.float32), dim=dim)
        out = torch.log(out)
        return out.to(dtype)

    def log_softmax_backward_impl(grad_output, output, dim, input_dtype):
        exp_output = torch.exp(output)
        grad_input = grad_output - exp_output * grad_output.sum(dim=dim, keepdim=True)
        return grad_input.to(input_dtype)

    lib.impl("_log_softmax", log_softmax_impl, "PrivateUse1")
    lib.impl("_log_softmax_backward_data", log_softmax_backward_impl, "PrivateUse1")

    return lib  # prevent GC


# Hold reference to prevent garbage collection of the library
_composite_ops_lib = None


def get_registered_ops():
    """Return list of registered FlagGems operators for flagos device."""
    return list(_registered_ops)


def is_flaggems_enabled():
    """Check if FlagGems operators are registered for flagos device."""
    return len(_registered_ops) > 0


# Auto-register FlagGems operators on import
_register_flaggems_operators()
_composite_ops_lib = _register_composite_ops()


# Re-export integration utilities
from torch_fl.integration import (  # noqa: E402
    is_flaggems_available,
    enable_flaggems_for_flagos,
    use_flaggems,
)

__all__ = [
    "flagos",
    "distributed",
    "get_registered_ops",
    "is_flaggems_enabled",
    "is_flaggems_available",
    "enable_flaggems_for_flagos",
    "use_flaggems",
]
