"""The torch.compile backend: partition, compile, and splice BPU calls into the graph.

Each BPU partition is replaced by one call_function node invoking a BPURuntime,
so the surrounding CPU nodes keep running in eager mode and control flow that
Dynamo already split out is untouched.
"""

from __future__ import annotations

import contextvars
import logging
import operator
from typing import Any, Callable

import torch
from torch.fx import GraphModule, Node

from .compiler import CompileError, compile_partition, find_hbdk
from .partition import (
    Partition,
    extract_subgraph,
    partition_graph,
    runtime_inputs,
    summarize,
)
from .runtime import BPURuntime

log = logging.getLogger("torch_fl.bpu")

# Dynamo's own graph inputs, stashed so the inner aten compiler can tell which
# of them are parameters and buffers. By the time AOTAutograd calls us the
# names are gone and the tensors are fake, so the information has to be carried
# across rather than recovered. A ContextVar keeps concurrent compiles apart.
_OUTER_INPUTS: contextvars.ContextVar[tuple[list[str], list[torch.Tensor]] | None] = (
    contextvars.ContextVar("torch_fl_bpu_outer_inputs", default=None)
)


class _BPUCall(torch.nn.Module):
    """Holds a BPURuntime so it survives as a graph attribute."""

    def __init__(self, rt: BPURuntime, n_outputs: int):
        super().__init__()
        self.rt = rt
        self.n_outputs = n_outputs

    def forward(self, *args: torch.Tensor):
        outs = self.rt(*args)
        return outs[0] if self.n_outputs == 1 else tuple(outs)


def _example_inputs_for(
    p: Partition,
    example_inputs: list[torch.Tensor],
    gm: GraphModule,
    frozen: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor] | None:
    """Materialize concrete tensors for a partition's boundary inputs.

    Uses the fake tensors Dynamo recorded in node.meta['val'] to synthesize
    real tensors of the right shape and dtype. Frozen weights use their real
    values, since the compiler bakes those into the artifact; everything else
    only needs the right shape.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    frozen = frozen or {}
    out = []
    # A backend runs under FakeTensorMode, so torch.zeros would itself produce
    # a fake tensor. The ONNX exporter needs real storage.
    with unset_fake_temporarily():
        for n in p.inputs:
            if n.name in frozen:
                continue
            val = n.meta.get("val")
            if not isinstance(val, torch.Tensor):
                return None
            if any(not isinstance(d, int) for d in val.shape):
                return None
            out.append(torch.zeros(tuple(val.shape), dtype=val.dtype))
    return out


# Dynamo names lifted module state after its access path, e.g.
# `l_self_modules_c1_parameters_weight_`. AOTAutograd renames these to
# `primals_N` but records the original index in node.meta['desc'].idx, which is
# how a weight is recognised two layers down from where it was named.
_STATE_MARKERS = ("_parameters_", "_buffers_")


def _frozen_weights(
    gm: GraphModule, example_inputs: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Map aten placeholder name -> constant tensor, for parameters and buffers.

    Returns {} when the mapping cannot be established, in which case weights
    stay as runtime inputs and the graph is still correct, just slower.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    outer = _OUTER_INPUTS.get()
    if not outer:
        return {}
    names, values = outer
    frozen: dict[str, torch.Tensor] = {}

    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        idx = getattr(node.meta.get("desc"), "idx", None)
        if idx is None or not (0 <= idx < len(values)):
            continue
        t, name = values[idx], names[idx]
        is_state = isinstance(t, torch.nn.Parameter) or any(
            m in name for m in _STATE_MARKERS
        )
        if is_state and isinstance(t, torch.Tensor) and not isinstance(t, FakeTensor):
            frozen[node.name] = t
    return frozen


def _splice(
    gm: GraphModule,
    p: Partition,
    mod: _BPUCall,
    tag: str,
    call_inputs: list[Node] | None = None,
) -> None:
    """Replace a partition's nodes with a single call into `mod`.

    `call_inputs` must match the compiled artifact's input order — frozen
    weights are baked in and are not passed.
    """
    setattr(gm, tag, mod)
    args = tuple(p.inputs if call_inputs is None else call_inputs)

    with gm.graph.inserting_before(p.nodes[0]):
        call = gm.graph.call_module(tag, args=args)

    if len(p.outputs) == 1:
        p.outputs[0].replace_all_uses_with(call)
    else:
        for i, old in enumerate(p.outputs):
            with gm.graph.inserting_after(call):
                item = gm.graph.call_function(operator.getitem, (call, i))
            old.replace_all_uses_with(item)

    for node in reversed(p.nodes):
        if not node.users:
            gm.graph.erase_node(node)


def bpu_backend(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    min_nodes: int = 3,
    strict: bool = False,
    act_scales: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Dynamo backend that offloads compilable subgraphs to the BPU.

    Dynamo hands us a graph of torch-level calls; AOTAutograd lowers it to
    aten ops with fake-tensor metadata, which is what the partitioner matches
    against.

    Partitions that cannot be compiled stay in the graph and run on CPU, so a
    missing or failing hbdk4 degrades performance but never correctness. Set
    `strict=True` to raise instead.

    `act_scales` maps ONNX tensor name to quantization scale (see
    `calibrate.calibrate_onnx`). Anything not listed falls back to
    `compiler.ACT_SCALE`.
    """
    from torch._dynamo.backends.common import aot_autograd

    def _compile_aten(aten_gm: GraphModule, aten_inputs: list[torch.Tensor]):
        return _offload(
            aten_gm,
            aten_inputs,
            min_nodes=min_nodes,
            strict=strict,
            act_scales=act_scales,
        )

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    token = _OUTER_INPUTS.set((names, list(example_inputs)))
    try:
        return aot_autograd(fw_compiler=_compile_aten)(gm, example_inputs)
    finally:
        _OUTER_INPUTS.reset(token)


def _offload(
    gm: GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    min_nodes: int = 3,
    strict: bool = False,
    act_scales: dict[str, float] | None = None,
) -> Callable[..., Any]:
    """Partition an aten graph and splice in BPU calls where possible."""
    partitions = partition_graph(gm, min_nodes=min_nodes)

    if not partitions:
        log.info("no BPU-eligible partitions; running on CPU")
        return gm.forward

    log.info("%s", summarize(gm, partitions))

    if find_hbdk() is None:
        msg = (
            "no hbdk4 compiler reachable — %d partition(s) will run on CPU. "
            "hbdk4 ships x86_64-only wheels, so it runs on this board under "
            "box64. Run scripts/setup_bpu_hbdk4.sh once, then set "
            "FLAGOS_BPU_X86_PYTHON and FLAGOS_BPU_X86_EMULATOR. The stock "
            "64 KB-page kernel is fine (box64 0.4+); see docs/bpu.md."
        )
        if strict:
            raise CompileError(msg % len(partitions))
        log.warning(msg, len(partitions))
        return gm.forward

    frozen = _frozen_weights(gm, example_inputs)
    if frozen:
        log.info("freezing %d weight tensor(s) into the artifact", len(frozen))
    else:
        log.warning(
            "could not identify parameters; weights will cross the boundary on "
            "every call, which usually costs more than the offload saves"
        )

    compiled = 0
    # Splice in reverse so earlier partitions' node references stay valid.
    for i, p in reversed(list(enumerate(partitions))):
        ex = _example_inputs_for(p, example_inputs, gm, frozen)
        if ex is None:
            log.warning("partition %d: dynamic shapes, keeping on CPU", i)
            continue
        try:
            sub = extract_subgraph(gm, p, frozen)
            hbm, ins, outs = compile_partition(sub, ex, act_scales=act_scales)
            rt = BPURuntime(str(hbm), ins, outs)
            _splice(
                gm,
                p,
                _BPUCall(rt, len(p.outputs)),
                f"_bpu_{i}",
                runtime_inputs(p, frozen),
            )
            compiled += 1
        except CompileError as e:
            if strict:
                raise
            log.warning("partition %d: %s — keeping on CPU", i, e)
        except Exception as e:  # noqa: BLE001 - never break the user's model
            if strict:
                raise
            log.warning("partition %d: unexpected %s: %s", i, type(e).__name__, e)

    if compiled:
        gm.graph.lint()
        gm.recompile()
        log.info("offloaded %d/%d partition(s) to BPU", compiled, len(partitions))

    return gm.forward


def register() -> None:
    """Register the backend so torch.compile(backend="bpu") resolves."""
    from torch._dynamo import register_backend

    register_backend(name="bpu", compiler_fn=bpu_backend)
