import os
import sys


def _select_backend_config() -> None:
    """Pick the op-routing config file based on the FLAGOS_USE_FLAGGEMS switch.

    The C++ dispatcher (csrc/aten/common.cc) reads FLAGOS_BACKEND_CONFIG to
    decide, per op, whether to run the CUDA boxing kernel or the FlagGems
    Python-path kernel. Both kernel sets are compiled into the wheel, so the
    choice is purely runtime:

      * FLAGOS_USE_FLAGGEMS=1  -> backends_flaggems.conf (FlagGems where available)
      * unset / 0              -> backends_cuda.conf      (pure CUDA)

    An explicit FLAGOS_BACKEND_CONFIG always wins (advanced/testing use), and
    the per-op FLAGOS_OP_<name> overrides in common.cc still apply on top. This
    must run before the first op dispatch triggers BackendTable() init; setting
    it at import time (before any flagos tensor op) is well before that.
    """
    if os.environ.get("FLAGOS_BACKEND_CONFIG"):
        return
    use_flaggems = os.environ.get("FLAGOS_USE_FLAGGEMS", "0") not in (
        "0",
        "",
        "off",
        "OFF",
        "false",
        "FALSE",
    )
    conf_name = "backends_flaggems.conf" if use_flaggems else "backends_cuda.conf"
    conf_path = os.path.join(os.path.dirname(__file__), conf_name)
    if os.path.exists(conf_path):
        os.environ["FLAGOS_BACKEND_CONFIG"] = conf_path


_select_backend_config()

# Optional: PyTorch wheels may require libcudart.so.12 version tags on MetaX.
if os.environ.get("FLAGOS_METAX_CUDART_SHIM", "0") == "1":
    from torch_fl.accelerator.metax._metax_cudart_shim import ensure_cudart_shim

    ensure_cudart_shim()

# When reusing PyTorch's CUDA boxing kernels on MetaX with a stock +cpu torch
# wheel, the active wheel's torch/lib must point at the MetaX C++ runtime .so.
# This MUST run before `import torch` (afterwards libc10 is already mapped and
# relinking is too late).  Gated on FLAGOS_METAX_BOXING=1; idempotent; no-op when
# torch already IS the MetaX wheel.
if os.environ.get("FLAGOS_METAX_BOXING", "0") == "1":
    from torch_fl.accelerator.metax._metax_libtorch_link import (
        ensure_maca_libtorch_links,
    )

    ensure_maca_libtorch_links()


def _preload_cuda_assets() -> None:
    """Load the bundled CUDA .so into this process BEFORE `import torch`.

    Hard constraint (docs/cpu_torch_external_libtorch_cuda.md §约束1): PyTorch
    caches its CUDAHooks on first `import torch`. If libtorch_cuda.so is loaded
    afterwards, device init fails with "Cannot initialize CUDA without ATen_cuda
    library" even though the kernels register. So we ctypes-dlopen it here, at
    the very top of torch_fl, before torch is imported.

    libtorch_cuda.so has unresolved deps on the NVIDIA runtime libs (libcudart,
    libcublas, libcudnn, libnvshmem_host, ...) shipped by the pip nvidia-*-cu12
    wheels. Since the process is already running, LD_LIBRARY_PATH cannot help;
    we must explicitly dlopen those deps (RTLD_GLOBAL) in dependency order first,
    then torch's own libc10/libtorch_cpu, then the CUDA libs.

    Skipped when:
      * FLAGOS_DISABLE_CUDA_ASSETS=1 (Ascend/MetaX/pure-CPU, or external preload)
      * the bundled libtorch_cuda.so is absent (e.g. slim build)
    """
    import ctypes
    import glob
    import importlib.util

    if os.environ.get("FLAGOS_DISABLE_CUDA_ASSETS", "0") == "1":
        return

    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    main_cuda = os.path.join(lib_dir, "libtorch_cuda.so")
    if not os.path.exists(main_cuda):
        # No bundled assets; rely on an out-of-band preload (e.g. LD_PRELOAD via
        # scripts/with_cuda_libtorch.sh) if the user set one up.
        return

    def _try(path, mode=ctypes.RTLD_GLOBAL):
        try:
            ctypes.CDLL(path, mode=mode)
            return True
        except OSError:
            return False

    # 1) NVIDIA runtime deps from pip nvidia-*-cu12 wheels. Locate their lib dirs
    #    via the installed `nvidia` namespace package (no torch import needed).
    nvidia_lib_dirs = []
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            nvidia_lib_dirs.extend(sorted(glob.glob(os.path.join(base, "*", "lib"))))
    # Dependency order: cudart first (everything needs it), then the math/comm
    # libs, then nvshmem. Load by soname glob; ignore any that are absent.
    _dep_order = [
        "libcudart.so*",
        "libnvrtc.so*",
        "libnvjitlink.so*",
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn*.so*",
        "libcufft.so*",
        "libcurand.so*",
        "libcusparse.so*",
        "libcusparseLt.so*",
        "libcusolver.so*",
        "libnccl.so*",
        "libnvshmem_host.so*",
        "libnvToolsExt.so*",
        "libcupti.so*",
    ]
    for pattern in _dep_order:
        for d in nvidia_lib_dirs:
            for so in sorted(glob.glob(os.path.join(d, pattern))):
                _try(so)

    # 2) torch's own CPU libs (libtorch_cuda depends on libc10 / libtorch_cpu).
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None and torch_spec.submodule_search_locations:
        torch_lib = os.path.join(
            list(torch_spec.submodule_search_locations)[0], "lib"
        )
        for name in ("libc10.so", "libtorch_cpu.so"):
            _try(os.path.join(torch_lib, name))

    # 3) Bundled CUDA libs. Order: nvshmem/nvrtc helpers, libc10_cuda, then the
    #    big libtorch_cuda.so (which pulls linalg on demand via bare dlopen, so
    #    its dir must be resolvable -- it is, since we load from lib_dir).
    for name in (
        "libtorch_nvshmem.so",
        "libcaffe2_nvrtc.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
        # linalg ops dlopen this by bare soname on demand; preloading makes the
        # loaded copy satisfy that later bare-name dlopen.
        "libtorch_cuda_linalg.so",
    ):
        p = os.path.join(lib_dir, name)
        if os.path.exists(p):
            _try(p)


_preload_cuda_assets()

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


# Expose libtorch symbols globally so triton-ascend's JIT-compiled launcher .so
# can resolve c10/ATen symbols (it links implicitly, not via DT_NEEDED).
import ctypes  # noqa: E402
import os as _os  # noqa: E402

_torch_lib = _os.path.join(_os.path.dirname(torch.__file__), "lib")
for _lib in ("libc10.so", "libtorch.so", "libtorch_cpu.so"):
    _p = _os.path.join(_torch_lib, _lib)
    if _os.path.exists(_p):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)

# Load libstream_api.so with RTLD_GLOBAL so that liboperators.so (FlagGems)
# can resolve GetCurrentStream at runtime.
_stream_api_path = _os.path.join(_os.path.dirname(__file__), "lib", "libstream_api.so")
if _os.path.exists(_stream_api_path):
    ctypes.CDLL(_stream_api_path, mode=ctypes.RTLD_GLOBAL)

import torch_fl._C  # type: ignore[misc]  # noqa: E402, F401
from . import flagos  # noqa: E402


torch.utils.rename_privateuse1_backend("flagos")
torch._register_device_module("flagos", flagos)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)

# Enable swap_tensors in Module._apply so that .to("flagos") preserves weight
# tying.  Without this, _apply creates new Parameter objects for PrivateUse1
# tensors (since _has_compatible_shallow_copy_type returns False for cross-device),
# breaking shared-storage relationships like lm_head.weight ↔ embed_tokens.weight.
# swap_tensors modifies the Parameter in-place, keeping object identity intact.
torch.__future__.set_swap_module_params_on_conversion(True)


# Global library instance to keep registrations alive
_flaggems_lib = None
_autograd_lib = None
_registered_ops = []


def _patch_flaggems_codegen_config():
    """
    Configure FlagGems' vendor + torch.cuda shim for the flagos device.

    FlagGems uses GEMS_VENDOR env var to detect the hardware vendor.

    - Generic NVIDIA CUDA (default when a real NVIDIA GPU is reachable via
      libcuda.so and MetaX compat is not requested): set GEMS_VENDOR=nvidia and
      shim torch.cuda so FlagGems' Triton kernels can compile/run under CPU
      torch + external libtorch_cuda.so. GEMS_VENDOR=nvidia is REQUIRED so
      FlagGems' tl_extra_shim resolves triton.language.extra.cuda.libdevice
      (which has `pow`); otherwise it falls back to tl.math (no `pow`).
      Disable with FLAGOS_DISABLE_CUDA_SHIM=1.

    - Ascend (fallback): set GEMS_VENDOR=ascend so FlagGems uses the ASCEND
      codegen config (prefer_block_pointer=False, avoiding a triton-ascend
      tl.make_block_ptr bug), and register torch.flagos as a torch.npu shim so
      FlagGems' gen_torch_device_object('ascend') resolves correctly.
    """
    import os
    import sys

    # --- Generic NVIDIA CUDA branch (default) ---
    if (
        os.environ.get("FLAGOS_DISABLE_CUDA_SHIM", "0") != "1"
        and os.environ.get("FLAGOS_METAX_COMPAT", "0") != "1"
        and os.environ.get("GEMS_VENDOR") != "ascend"
    ):
        from torch_fl.accelerator.cuda._cuda_compat import (
            is_nvidia_cuda_available,
            patch_torch_cuda_for_flagos,
        )

        if is_nvidia_cuda_available():
            os.environ.setdefault("GEMS_VENDOR", "nvidia")
            patch_torch_cuda_for_flagos()
            return

    # --- Ascend fallback branch ---
    # Set vendor before FlagGems runtime initializes
    if "GEMS_VENDOR" not in os.environ:
        os.environ["GEMS_VENDOR"] = "ascend"

    # FlagGems' ASCEND backend expects torch.npu to exist (device_name="npu").
    # Provide torch.flagos as a shim so gen_torch_device_object() succeeds.
    # Mark is_available()=False so transformers/accelerate don't think real
    # NPU hardware is present and try to import npu_fusion_attention etc.
    if not hasattr(torch, "npu"):
        import types

        _npu_device_shim = types.ModuleType("torch.npu")
        _npu_device_shim.is_available = lambda: False
        _npu_device_shim.device_count = flagos.device_count
        _npu_device_shim.current_device = flagos.current_device
        _npu_device_shim.set_device = flagos.set_device
        _npu_device_shim.synchronize = flagos.synchronize
        _npu_device_shim.device = flagos.device
        _npu_device_shim.Stream = flagos.Stream
        _npu_device_shim.Event = flagos.Event
        _npu_device_shim.current_stream = flagos.current_stream
        _npu_device_shim.default_generators = flagos.default_generators
        torch.npu = _npu_device_shim

    # FlagGems' ASCEND backend imports torch_npu in _get_vendor_from_quick_cmd.
    # Provide a minimal shim module so the import doesn't fail.
    # Also set __spec__ to satisfy importlib.util.find_spec() checks (used by
    # accelerate.utils.imports.is_npu_available).
    if "torch_npu" not in sys.modules:
        import types
        import importlib.machinery

        _npu_shim = types.ModuleType("torch_npu")
        _npu_shim.npu = _npu_device_shim
        _npu_shim.__spec__ = importlib.machinery.ModuleSpec(
            name="torch_npu",
            loader=None,
            origin="torch_fl_shim",
        )
        sys.modules["torch_npu"] = _npu_shim


# Patch FlagGems codegen config before any FlagGems code is imported
_patch_flaggems_codegen_config()


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
    # log_softmax - registered in C++ with CUDA structured kernels
    "_log_softmax",
    "_log_softmax_backward_data",
    "_softmax_backward_data",
    "div.Scalar",
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

    Disabled: Python-layer FlagGems registration is not used.
    All ops are dispatched through the C++ stub path instead.
    """
    global _flaggems_lib, _autograd_lib, _registered_ops

    if os.environ.get("FLAGOS_DISABLE_FLAGGEMS_PY", "0") == "1":
        _registered_ops = []
        return 0

    import importlib.util

    if importlib.util.find_spec("flag_gems") is None:
        # flag_gems not installed, will use cpu_fallback
        return 0

    _flaggems_lib = torch.library.Library("aten", "IMPL")
    _registered_ops = []
    return 0


def _register_composite_ops():
    """
    Register CompositeExplicitAutograd ops that cause cpu_fallback segfault.

    Some PyTorch ops are CompositeExplicitAutograd (not CompositeImplicitAutograd),
    meaning they don't auto-decompose for PrivateUse1. They fall through to
    cpu_fallback which segfaults when handling privateuseone tensors.

    Previously _log_softmax and _log_softmax_backward_data were registered here
    as Python decompositions. They are now registered in C++ with proper CUDA
    structured kernels for full performance.
    """
    lib = torch.library.Library("aten", "IMPL")

    # No Python-registered ops remaining; keep the library alive for future use.

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
