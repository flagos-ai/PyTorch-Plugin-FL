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
from torch._C import _distributed_c10d as _c10d


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
        self._store = store
        self._timeout = timeout
        self._view_fn = self._build_inner(store, rank, world_size, timeout)

    def _build_inner(self, store, rank, world_size, timeout):
        """Create the inner backend and return the view-conversion function.

        Priority: FlagCX (heterogeneous) -> HCCL (ascend) -> NCCL (nvidia/metax).
        The returned callable maps a privateuseone tensor to the device view the
        inner backend expects (or is unused when the backend handles flagos
        tensors natively).
        """
        import torch_fl._C as _C

        vendor = os.environ.get("GEMS_VENDOR", "nvidia")

        # --- Try FlagCX first (heterogeneous unified comm) ---
        # FlagCX self-registers the "flagcx" backend for the "cuda" device on
        # ``import flagcx`` (via Backend.register_backend, extended_api=True).
        # Its ProcessGroup is created through createFlagcxBackend(opts, extra),
        # NOT a plain (store, rank, world_size) ctor, so we build the
        # _DistributedBackendOptions the same way c10d does.
        if self._try_build_flagcx(store, rank, world_size, timeout):
            # FlagCX operates on cuda tensors; flagos shares physical GPU
            # memory with cuda, so hand it a zero-copy cuda view.
            # TODO: if a future FlagCX adaptor accepts privateuseone directly,
            # set self._needs_view = False and return an identity view here.
            return _C._flagos_to_cuda_view

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

    def _try_build_flagcx(self, store, rank, world_size, timeout) -> bool:
        """Instantiate a FlagCX inner backend if flagcx is importable.

        Returns True and sets ``self._inner`` / ``self._needs_view`` on success,
        False if flagcx is unavailable. Any hard failure is surfaced as a
        warning and treated as unavailable so we fall through to NCCL/HCCL.
        """
        try:
            import flagcx  # noqa: F401 — self-registers "flagcx" backend
        except ImportError:
            return False

        # createFlagcxBackend is the extended_api creator exposed by flagcx._C.
        creator = getattr(flagcx, "createFlagcxBackend", None)
        if creator is None:
            # older builds only expose the class; skip and fall back
            warnings.warn("[ProcessGroupFlagOS] flagcx present but "
                          "createFlagcxBackend missing; falling back.")
            return False

        try:
            from torch._C._distributed_c10d import _DistributedBackendOptions
            opts = _DistributedBackendOptions()
            opts.store = store
            opts.group_rank = rank
            opts.group_size = world_size
            try:
                opts.group_id = self.group_name or ""
            except Exception:
                opts.group_id = ""
            opts.global_ranks_in_group = list(range(world_size))
            if timeout is not None:
                opts.timeout = timeout

            # extra Options: enable_tuner / tune_group_idx (see backend_flagcx.cpp).
            # ProcessGroupFlagCX lives on the flagcx module (flagcx._C), not on
            # torch.distributed.
            pg_cls = getattr(flagcx, "ProcessGroupFlagCX", None) or \
                getattr(torch.distributed, "ProcessGroupFlagCX", None)
            extra_cls = getattr(pg_cls, "Options", None)
            extra = extra_cls() if extra_cls is not None else None

            self._inner = creator(opts, extra) if extra is not None else creator(opts)
            self._needs_view = True
            return True
        except Exception as e:
            warnings.warn(f"[ProcessGroupFlagOS] FlagCX init failed ({e}); "
                          f"falling back to vendor-native backend.")
            return False

    # ------------------------------------------------------------------
    # Collective virtuals
    #
    # Convention:
    #   _tl(lst)  → convert List[Tensor]
    #   _tll(lst) → convert List[List[Tensor]]
    # Each method converts inputs, delegates to self._inner, returns Work.
    # ------------------------------------------------------------------

    # NOTE: opts default to None (constructed lazily). Options types are not all
    # exposed on ``torch.distributed`` (e.g. AllgatherOptions lives only in
    # ``torch._C._distributed_c10d``), and mutable defaults evaluated at def-time
    # would be shared singletons. c10d always passes opts explicitly when calling
    # these virtuals, so the fallback construction is just a safety net.

    def allreduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceOptions()
        return self._inner.allreduce(_tl(tensors, self._view_fn), opts)

    def allreduce_coalesced(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.AllreduceCoalescedOptions()
        return self._inner.allreduce_coalesced(_tl(tensors, self._view_fn), opts)

    def allgather(self, output_tensors, input_tensors, opts=None):
        # output_tensors: List[List[Tensor]], input_tensors: List[Tensor]
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def allgather_coalesced(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather_coalesced(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def allgather_into_tensor_coalesced(self, output_tensors, input_tensors,
                                        opts=None):
        if opts is None:
            opts = _c10d.AllgatherOptions()
        return self._inner.allgather_into_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def broadcast(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.BroadcastOptions()
        return self._inner.broadcast(_tl(tensors, self._view_fn), opts)

    def reduce(self, tensors, opts=None):
        if opts is None:
            opts = _c10d.ReduceOptions()
        return self._inner.reduce(_tl(tensors, self._view_fn), opts)

    def reduce_scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner.reduce_scatter(
            _tl(output_tensors, self._view_fn),
            _tll(input_tensors, self._view_fn),
            opts,
        )

    def reduce_scatter_tensor_coalesced(self, output_tensors, input_tensors,
                                        opts=None):
        if opts is None:
            opts = _c10d.ReduceScatterOptions()
        return self._inner.reduce_scatter_tensor_coalesced(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def alltoall(self, output_tensors, input_tensors, opts=None):
        if opts is None:
            opts = _c10d.AllToAllOptions()
        return self._inner.alltoall(
            _tl(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def alltoall_base(self, output_tensor, input_tensor,
                      output_split_sizes, input_split_sizes, opts=None):
        if opts is None:
            opts = _c10d.AllToAllOptions()
        return self._inner.alltoall_base(
            _to_comm(output_tensor, self._view_fn),
            _to_comm(input_tensor, self._view_fn),
            output_split_sizes, input_split_sizes,
            opts,
        )

    def gather(self, output_tensors, input_tensors, opts=None):
        # output: List[List[Tensor]], input: List[Tensor]
        if opts is None:
            opts = _c10d.GatherOptions()
        return self._inner.gather(
            _tll(output_tensors, self._view_fn),
            _tl(input_tensors, self._view_fn),
            opts,
        )

    def scatter(self, output_tensors, input_tensors, opts=None):
        # output: List[Tensor], input: List[List[Tensor]]
        if opts is None:
            opts = _c10d.ScatterOptions()
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

    def barrier(self, opts=None):
        # barrier carries no tensors; delegate directly
        if opts is None:
            opts = _c10d.BarrierOptions()
        return self._inner.barrier(opts)

    def monitored_barrier(self, opts=None, wait_all_ranks=False):
        if opts is None:
            opts = _c10d.BarrierOptions()
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
