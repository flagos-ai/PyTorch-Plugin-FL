import os
import sys

# Optional: PyTorch wheels may require libcudart.so.12 version tags on MetaX.
if os.environ.get("FLAGOS_METAX_CUDART_SHIM", "0") == "1":
    from torch_fl.accelerator.metax._metax_cudart_shim import ensure_cudart_shim

    ensure_cudart_shim()

import torch  # noqa: E402


if sys.platform == "win32":
    from ._utils import _load_dll_libraries

    _load_dll_libraries()
    del _load_dll_libraries


# Optional FlagGems-on-MetaX compat (does not patch torch.cuda unless enabled).
if os.environ.get("FLAGOS_METAX_COMPAT", "0") == "1":
    from torch_fl.accelerator.metax._metax_compat import (  # noqa: E402
        is_metax_available,
        patch_torch_cuda_for_metax,
    )

    if is_metax_available():
        patch_torch_cuda_for_metax()


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

# Initialize CUDA runtime only when FlagGems Python path needs it (CUDA backend ops).
if (
    os.environ.get("FLAGOS_DISABLE_FLAGGEMS_PY", "0") != "1"
    and torch.cuda.is_available()
):
    torch.cuda.init()


# Ops that use torch_device_fn.device(device) with explicit device parameter
# These don't work with flagos device and should use cpu_fallback instead
_EXCLUDED_OPS = {
    # Factory functions that take device parameter
    "randn",
    "randn_like",
    "rand",
    "rand_like",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "arange",
    "arange.start",
    "arange.start_step",
    "linspace",
    "logspace",
    "eye",
    "eye.m",
    "randperm",
    "empty.memory_format",  # Already registered in C++
    "empty_strided",  # Already registered in C++
    # Random ops that use device context
    "uniform_",
    "normal.float_Tensor",
    "normal.Tensor_float",
    "normal.Tensor_tensor",
    "exponential_",
    "multinomial",
    # Copy ops - already registered in C++, skip to avoid duplicate registration
    "copy_",
    "_to_copy",
    "contiguous",
    "clone",
    # log_softmax - FlagGems Triton kernel exceeds MetaX's 4KB/thread private memory
    # limit on large vocab (e.g. Qwen3 151k). Use Python decomposition instead.
    "_log_softmax",
    "_log_softmax_backward_data",
    # Ops dispatched by C++ stub (DispatchStub) which reads backends.conf
    # at load time to route to flaggems or cuda per-op.
    "mm",
    "mm.out",
    "bmm",
    "bmm.out",
    "cat",
    "embedding",
    "add.Tensor",
    "mul.Tensor",
    "silu",
    "rsqrt",
    "mean.dim",
    "cos",
    "sin",
    "neg",
    "pow.Tensor_Scalar",
    "all",
    "_softmax",
    "bitwise_and.Tensor",
    "le.Tensor",
    "where.self",
    "index.Tensor",
    "new_ones",
    "scalar_tensor",
    "ones_like",
    "zeros",
    "silu_backward",
    "sum.dim_IntList",
    "slice_backward",
    "constant_pad_nd",
    "embedding_dense_backward",
    "nll_loss_forward",
    "nll_loss_backward",
}


# Cache for CUDA runtime library
_cudart_lib = None
_cudaMemcpy = None


def _get_cudaMemcpy():
    """Get cudaMemcpy function from CUDA runtime library (cached)."""
    global _cudart_lib, _cudaMemcpy
    if _cudaMemcpy is not None:
        return _cudaMemcpy

    import ctypes

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

    Flagos and CUDA share the same GPU memory, so FlagGems Triton kernels
    can operate directly on flagos tensor pointers without conversion.

    Set FLAGOS_DISABLE_FLAGGEMS_PY=1 to skip Python-layer FlagGems registration,
    leaving only the C++ stub dispatch path active.
    """
    global _flaggems_lib, _autograd_lib, _registered_ops

    if os.environ.get("FLAGOS_DISABLE_FLAGGEMS_PY", "0") == "1":
        _registered_ops = []
        return 0

    try:
        from flag_gems import _FULL_CONFIG
    except ImportError:
        # flag_gems not installed, will use cpu_fallback
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []

    # Build mapping of backward ops
    backward_ops = {}
    for item in _FULL_CONFIG:
        if len(item) >= 2:
            op_name = item[0]
            if "backward" in op_name.lower():
                backward_ops[op_name] = item[1]

    for item in _FULL_CONFIG:
        if len(item) < 2:
            continue

        op_name = item[0]
        impl_func = item[1]

        # Skip excluded ops - they will use cpu_fallback
        if op_name in _EXCLUDED_OPS:
            continue

        # Check version conditions if present
        if len(item) > 2:
            condition = item[2]
            if callable(condition) and not condition():
                continue

        try:
            _flaggems_lib.impl(op_name, impl_func, "PrivateUse1")
            _registered_ops.append(op_name)
        except Exception:
            # Some operators may already be registered or have incompatible signatures
            pass

    return len(_registered_ops)


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
