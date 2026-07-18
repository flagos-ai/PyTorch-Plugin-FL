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
Distributed training utilities for flagos device.

Provides transparent integration between flagos (PrivateUse1) tensors and
PyTorch's distributed communication backends (NCCL, FlagCX, etc.).

Typical usage::

    import torch_fl
    import torch_fl.distributed as flagos_dist

    # Initialize process group — automatically registers backend for
    # privateuseone and patches dist collectives for flagos tensors.
    flagos_dist.init_process_group(backend="nccl")      # or "flagcx"

    # Wrap model with DDP — automatically uses python_reducer workaround.
    model = flagos_dist.DistributedDataParallel(model)

    # Everything else is standard PyTorch distributed training.
"""

import functools

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as _DDP


# ---------------------------------------------------------------------------
# Internal: patch dist collectives to handle flagos tensors
# ---------------------------------------------------------------------------

_collectives_patched = False


def _patch_dist_collectives():
    """Patch torch.distributed collectives to transparently handle flagos tensors.

    Communication backends (NCCL, FlagCX) operate on CUDA tensors. This patches
    the Python-level distributed functions to convert flagos (privateuseone)
    tensors to CUDA views before forwarding to the underlying backend.

    This is idempotent — calling it multiple times is safe.
    """
    global _collectives_patched
    if _collectives_patched:
        return
    _collectives_patched = True

    import torch_fl._C as _C

    def _ensure_cuda(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.device.type in (
            "privateuseone",
            "flagos",
        ):
            return _C._flagos_to_cuda_view(tensor)
        return tensor

    _orig_all_reduce = dist.all_reduce

    def _all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return _orig_all_reduce(
            _ensure_cuda(tensor), op=op, group=group, async_op=async_op
        )

    dist.all_reduce = _all_reduce

    _orig_broadcast = dist.broadcast

    def _broadcast(tensor, src, group=None, async_op=False):
        return _orig_broadcast(
            _ensure_cuda(tensor), src=src, group=group, async_op=async_op
        )

    dist.broadcast = _broadcast

    _orig_reduce = dist.reduce

    def _reduce(tensor, dst, op=dist.ReduceOp.SUM, group=None, async_op=False):
        return _orig_reduce(
            _ensure_cuda(tensor), dst=dst, op=op, group=group, async_op=async_op
        )

    dist.reduce = _reduce

    _orig_all_gather_into_tensor = dist.all_gather_into_tensor

    def _all_gather_into_tensor(output, input, group=None, async_op=False):
        return _orig_all_gather_into_tensor(
            _ensure_cuda(output),
            _ensure_cuda(input),
            group=group,
            async_op=async_op,
        )

    dist.all_gather_into_tensor = _all_gather_into_tensor

    _orig_reduce_scatter_tensor = dist.reduce_scatter_tensor

    def _reduce_scatter_tensor(
        output, input, op=dist.ReduceOp.SUM, group=None, async_op=False
    ):
        return _orig_reduce_scatter_tensor(
            _ensure_cuda(output),
            _ensure_cuda(input),
            op=op,
            group=group,
            async_op=async_op,
        )

    dist.reduce_scatter_tensor = _reduce_scatter_tensor


# ---------------------------------------------------------------------------
# Internal: register GPU backend for privateuseone device type
# ---------------------------------------------------------------------------


def _register_privateuseone_backend(comm):
    """Register the GPU communication backend for the privateuseone device type.

    After ``dist.init_process_group``, the backend is registered for ``cuda``
    but not for ``privateuseone`` (flagos). This copies the registration so that
    distributed operations on flagos tensors are routed to the same backend.

    Args:
        comm: Communication backend name — ``"nccl"`` or ``"flagcx"``.
    """
    from torch.distributed.distributed_c10d import ProcessGroup

    pg = dist.distributed_c10d._get_default_group()
    gpu_backend = pg._get_backend(torch.device("cuda"))
    backend_type = (
        ProcessGroup.BackendType.CUSTOM
        if comm == "flagcx"
        else ProcessGroup.BackendType.NCCL
    )
    pg._register_backend(torch.device("privateuseone"), backend_type, gpu_backend)


# ---------------------------------------------------------------------------
# Public API: init_process_group
# ---------------------------------------------------------------------------


def init_process_group(backend="nccl", **kwargs):
    """Initialize distributed process group for flagos device.

    Wraps ``torch.distributed.init_process_group`` and additionally:
    1. Registers the communication backend for the ``privateuseone`` device
       type so that distributed ops work on flagos tensors.
    2. Patches ``torch.distributed`` collectives (``all_reduce``, ``broadcast``,
       etc.) to transparently convert flagos tensors to CUDA views.

    Args:
        backend: ``"nccl"`` or ``"flagcx"``. When ``"flagcx"`` is specified,
            the actual PyTorch backend string becomes ``"cpu:gloo,cuda:flagcx"``.
        **kwargs: Forwarded to ``torch.distributed.init_process_group``.
    """
    if backend == "flagcx":
        import flagcx  # noqa: F401 — registers "flagcx" via torch.backends entry point

        dist.init_process_group(backend="cpu:gloo,cuda:flagcx", **kwargs)
    else:
        dist.init_process_group(backend=backend, **kwargs)

    _register_privateuseone_backend(comm=backend)
    # no need for FlagCX, but NCCL needs
    _patch_dist_collectives()


# ---------------------------------------------------------------------------
# Public API: DistributedDataParallel
# ---------------------------------------------------------------------------


class DistributedDataParallel(_DDP):
    """DDP wrapper for flagos models.

    PyTorch's C++ Reducer uses CUDA Futures which fail for privateuseone
    tensors. This wrapper transparently enables ``python_reducer`` mode and
    installs custom ``post_accumulate_grad_hook`` handlers that call the
    patched ``dist.all_reduce`` (which converts flagos→CUDA under the hood).

    Usage is identical to ``torch.nn.parallel.DistributedDataParallel``::

        model = flagos_dist.DistributedDataParallel(model)
    """

    def __init__(self, module, **kwargs):
        # Force python_reducer mode to avoid C++ CUDA Future validation
        import torch._dynamo.utils

        _orig = torch._dynamo.utils.get_optimize_ddp_mode
        torch._dynamo.utils.get_optimize_ddp_mode = lambda: "python_reducer"
        try:
            kwargs.setdefault("init_sync", False)
            kwargs.setdefault("broadcast_buffers", False)
            kwargs.setdefault("gradient_as_bucket_view", True)
            super().__init__(module, **kwargs)
        finally:
            torch._dynamo.utils.get_optimize_ddp_mode = _orig

        # Replace default accum_grad_hooks with flagos-compatible version
        for h in self._accum_grad_hooks:
            h.remove()
        self._accum_grad_hooks.clear()

        for param in self._module_parameters:
            if param.requires_grad:
                self._accum_grad_hooks.append(
                    param.register_post_accumulate_grad_hook(
                        functools.partial(self._accum_grad_hook, ddp_model=self)
                    )
                )

    @staticmethod
    def _accum_grad_hook(param, *, ddp_model):
        if not ddp_model.require_backward_grad_sync:
            return
        if param.grad is None:
            return
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(dist.get_world_size())


# ---------------------------------------------------------------------------
# Public API: move_buffers_to_device
# ---------------------------------------------------------------------------


def move_buffers_to_device(module, device):
    """Recursively move all module buffers to the specified device.

    When loading a model to the flagos device, some buffers (e.g. causal masks)
    may remain on CPU or CUDA. This ensures all buffers are on the correct
    device.

    Args:
        module: ``torch.nn.Module`` to process.
        device: Target device (e.g. ``"flagos:0"``).
    """
    for name, buf in module._buffers.items():
        if buf is not None and buf.device.type not in ("privateuseone", "flagos"):
            module._buffers[name] = buf.to(device)
    for child in module.children():
        move_buffers_to_device(child, device)


__all__ = [
    "init_process_group",
    "DistributedDataParallel",
    "move_buffers_to_device",
]
