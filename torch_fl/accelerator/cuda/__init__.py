"""Generic NVIDIA CUDA compatibility layer for the flagos backend."""

from ._cuda_compat import (
    is_nvidia_cuda_available,
    patch_torch_cuda_for_flagos,
)

__all__ = [
    "is_nvidia_cuda_available",
    "patch_torch_cuda_for_flagos",
]
