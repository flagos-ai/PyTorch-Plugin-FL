"""
ProcessGroupFlagOS — a transparent ProcessGroup backend for flagos (PrivateUse1).

Architecture
------------
flagos tensors share physical GPU memory with CUDA tensors. This ProcessGroup
wraps an underlying comm backend (FlagCX if available, otherwise NCCL/HCCL)
and converts privateuseone tensors to the backend's expected device view inside
each collective virtual method. The Work objects returned are from the inner
backend, so callers (including the DDP Reducer) get properly typed futures.

FlagCX path (no view conversion needed)
    When flagcx is installed and its adaptor natively handles privateuseone
    tensors, ``_needs_view = False`` and tensors are passed through unchanged.
    TODO: verify this assumption on a flagcx-enabled machine.

NCCL/HCCL path (view conversion)
    Privateuseone tensors are reinterpreted as CUDA tensors via a zero-copy
    shared-storage view (``_C._flagos_to_cuda_view``), then handed to NCCL.
    The underlying buffer is the same physical memory, so NCCL's write-back
    is visible to the privateuseone side immediately.
"""

import os
import warnings

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Tensor view helpers
# ---------------------------------------------------------------------------

def _to_comm(t: torch.Tensor, view_fn) -> torch.Tensor:
    """Convert a single tensor for the comm backend if it is on flagos."""
    if t.device.type in ("privateuseone", "flagos"):
        return view_fn(t)
    return t


def _tl(tensors, view_fn):
    """Convert a list of tensors."""
    return [_to_comm(t, view_fn) for t in tensors]


def _tll(tensor_lists, view_fn):
    """Convert a list-of-lists of tensors (used by allgather / reduce_scatter)."""
    return [_tl(tl, view_fn) for tl in tensor_lists]


# ---------------------------------------------------------------------------
# ProcessGroupFlagOS
# ---------------------------------------------------------------------------

class ProcessGroupFlagOS(dist.ProcessGroup):
    """ProcessGroup backend for flagos (PrivateUse1).

    Wraps an underlying NCCL or FlagCX ProcessGroup.  Each collective virtual
    method converts privateuseone tensors to the appropriate backend view before
    delegating; the inner backend's Work is returned directly.

    Instantiated by the ``creator_fn`` registered via
    ``dist.Backend.register_backend``.  Do not instantiate directly.
    """

    def __init__(self, store, rank: int, world_size: int, timeout=None):
        super().__init__(rank, world_size)
        self._view_fn = self._build_inner(store, rank, world_size, timeout)

    def _build_inner(self, store, rank, world_size, timeout):
        """Create the inner backend and return the view-conversion function.

        Returns the view function (or identity if no conversion is needed).
        """
        import torch_fl._C as _C

        vendor = os.environ.get("GEMS_VENDOR", "nvidia")

        # --- Try FlagCX first (heterogeneous unified comm) ---
        flagcx_pg_cls = getattr(torch.distributed, "ProcessGroupFlagCX", None)
        if flagcx_pg_cls is not None:
            try:
                opts = flagcx_pg_cls.Options()
                self._inner = flagcx_pg_cls(store, rank, world_size, opts)
                # TODO: verify on a flagcx machine whether privateuseone
                # tensors are accepted natively. Set _needs_view=False if so.
                self._needs_view = True
                return _C._flagos_to_cuda_view
            except Exception as e:
                warnings.warn(f"[ProcessGroupFlagOS] FlagCX init failed ({e}); falling back.")

        # --- Ascend: HCCL via torch_npu ---
        if vendor == "ascend":
            try:
                import torch_npu.distributed  # noqa
                hccl_cls = getattr(torch.distributed, "ProcessGroupHCCL", None)
                if hccl_cls is None:
                    hccl_cls = getattr(torch_npu.distributed, "ProcessGroupHCCL", None)
                if hccl_cls is not None:
                    self._inner = hccl_cls(store, rank, world_size)
                    self._needs_view = True
                    # TODO: implement _flagos_to_npu_view in csrc/module.cc
                    if not hasattr(_C, "_flagos_to_npu_view"):
                        raise NotImplementedError(
                            "_flagos_to_npu_view not yet implemented. "
                            "Use FlagCX on Ascend to avoid this path."
                        )
                    return _C._flagos_to_npu_view
            except ImportError:
                warnings.warn("[ProcessGroupFlagOS] torch_npu not found; cannot use HCCL.")

        # --- NCCL fallback (nvidia / metax) ---
        nccl_cls = getattr(torch.distributed, "ProcessGroupNCCL", None)
        if nccl_cls is None:
            raise RuntimeError(
                "ProcessGroupFlagOS: no suitable inner backend found. "
                "Install flagcx, or ensure PyTorch was built with NCCL support."
            )
        opts = nccl_cls.Options()
        if timeout is not None:
            opts._timeout = timeout
        self._inner = nccl_cls(store, rank, world_size, opts)
        self._needs_view = True
        return _C._flagos_to_cuda_view

    # ------------------------------------------------------------------
    # Collective virtuals
    #
    # Convention:
    #   _tl(lst)  → convert List[Tensor]
    #   _tll(lst) → convert List[List[Tensor]]
    # Each method converts inputs, delegates to self._inner, returns Work.
    # ------------------------------------------------------------------

    def allreduce(self, tensors, opts=dist.AllreduceOptions()):
        return self._inner.allreduce(_tl(tensors, self._view_fn), opts)

    def allreduce_coalesced(self, tensors, opts=dist.AllreduceCoalescedOptions()):
        return self._inner.allreduce_coalesced(_tl(tensors, self._view_fn), opts)

    def allgather(self, output_tensors, input_tensors,
                  opts=dist.AllgatherOptions()):
        # output_tensors: List[List[Tensor]], input_tensors: List[Tensor]
        return self._inner.allgather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def allgather_coalesced(self, output_tensors, input_tensors,
                            opts=dist.AllgatherOptions()):
        return self._inner.allgather_coalesced(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def allgather_into_tensor_coalesced(self, output_tensors, input_tensors,
                                        opts=dist.AllgatherOptions()):
        return self._inner.allgather_into_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def broadcast(self, tensors, opts=dist.BroadcastOptions()):
        return self._inner.broadcast(_tl(tensors, self._view_fn), opts)

    def reduce(self, tensors, opts=dist.ReduceOptions()):
        return self._inner.reduce(_tl(tensors, self._view_fn), opts)

    def reduce_scatter(self, output_tensors, input_tensors,
                       opts=dist.ReduceScatterOptions()):
        # output: List[Tensor], input: List[List[Tensor]]
        return self._inner.reduce_scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors,
                                        opts=dist.ReduceScatterOptions()):
        return self._inner.reduce_scatter_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def alltoall(self, output_tensors, input_tensors,
                 opts=dist.AllToAllOptions()):
        return self._inner.alltoall(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def alltoall_base(self, output_tensor, input_tensor,
                      output_split_sizes, input_split_sizes,
                      opts=dist.AllToAllOptions()):
        return self._inner.alltoall_base(
            _to_comm(output_tensor, self._view_fn),
            _to_comm(input_tensor, self._view_fn),
            output_split_sizes, input_split_sizes,
            opts,
        )

    def gather(self, output_tensors, input_tensors,
               opts=dist.GatherOptions()):
        # output: List[List[Tensor]], input: List[Tensor]
        return self._inner.gather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def scatter(self, output_tensors, input_tensors,
                opts=dist.ScatterOptions()):
        # output: List[Tensor], input: List[List[Tensor]]
        return self._inner.scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    def send(self, tensors, dst_rank: int, tag: int):
        return self._inner.send(_tl(tensors, self._view_fn), dst_rank, tag)

    def recv(self, tensors, src_rank: int, tag: int):
        return self._inner.recv(_tl(tensors, self._view_fn), src_rank, tag)

    def recv_anysource(self, tensors, tag: int):
        return self._inner.recv_anysource(_tl(tensors, self._view_fn), tag)

    def barrier(self, opts=dist.BarrierOptions()):
        # barrier carries no tensors; delegate directly
        return self._inner.barrier(opts)

    def monitored_barrier(self, opts=dist.BarrierOptions(),
                          wait_all_ranks=False):
        return self._inner.monitored_barrier(opts, wait_all_ranks)


# ---------------------------------------------------------------------------
# Backend creator function + public registration helper
# ---------------------------------------------------------------------------

def _create_flagos_pg(store, rank, world_size, timeout):
    """Creator function called by torch.distributed._new_process_group_helper."""
    return ProcessGroupFlagOS(store, rank, world_size, timeout=timeout)


def register_flagos_backend() -> None:
    """Register the ``"flagos"`` backend and set it as default for privateuseone.

    Called once at ``import torch_fl``.  Subsequent calls are no-ops.
    """
    if "flagos" in dist.Backend.backend_list:
        return  # already registered

    dist.Backend.register_backend(
        "flagos",
        _create_flagos_pg,
        extended_api=False,
        devices=["privateuseone"],
    )

    # Make `init_process_group(device_id=torch.device("privateuseone:0"))`
    # auto-select "flagos" without the user specifying a backend string.
    dist.Backend.default_device_backend_map.setdefault("privateuseone", "flagos")
