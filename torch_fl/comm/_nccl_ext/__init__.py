"""flagos NCCL backend extension.

Exposes c10d::ProcessGroupNCCL on CPU-only torch builds that ship without
USE_C10D_NCCL, by linking against an externally preloaded libtorch_cuda.so.
The compiled module (_flagos_nccl) is built out-of-tree via build.py; it may be
absent in slim installs, in which case ProcessGroupFlagOS falls back to FlagCX.
"""

try:
    from . import _flagos_nccl  # noqa: F401
except ImportError:  # pragma: no cover - extension not built
    _flagos_nccl = None
