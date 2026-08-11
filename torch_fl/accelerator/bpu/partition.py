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

"""FX graph partitioning: split a graph into BPU-compilable subgraphs and CPU remainder.

The BPU only executes whole compiled graphs, so the unit of offload is a maximal
connected region of supported nodes rather than an individual operator.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.fx import Graph, GraphModule, Node

# Operators hbdk4 can compile. Conservative on purpose: an op that is wrongly
# listed here fails at compile time (recoverable, we fall back), whereas a
# missing op only costs us a smaller subgraph.
_SUPPORTED: set[Callable | str] = {
    # convolution / linear
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    # normalization
    torch.ops.aten._native_batch_norm_legit_no_training.default,
    torch.ops.aten.batch_norm.default,
    torch.ops.aten.native_layer_norm.default,
    # activation
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardswish.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.gelu.default,
    # pooling
    torch.ops.aten.max_pool2d.default,
    # max_pool2d_with_indices is deliberately excluded: it returns an indices
    # tensor the BPU does not produce, and the decomposition is what Dynamo
    # actually emits. Use aten.max_pool2d via a graph pass if you need pooling
    # inside a partition.
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.mean.dim,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten._adaptive_avg_pool2d.default,
    # elementwise
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.mul_.Tensor,
    # RMSNorm, which every recent LLM uses in place of LayerNorm. hbdk4 lowers
    # the lot to b30vpu (`pow` -> Pow, `rsqrt` -> Sqrt+Div in ONNX): checked
    # with convert(advice=True), 6 ops, zero CPU fallbacks. Without these two a
    # transformer block splits at every norm -- 86% coverage and two partitions
    # where one will do.
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.neg.default,
    # shape
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.cat.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.unsqueeze.default,
    # `slice` is how attention takes its causal window and how any model that
    # indexes a cache reads it. Without it a transformer fragments at every
    # slice; with it decode and prefill are each a single partition.
    torch.ops.aten.slice.Tensor,
    # `expand` is what AOTAutograd inserts ahead of a batched matmul to
    # broadcast the operands; ONNX Expand covers it exactly. `_unsafe_view`
    # and `t` are rewritten to `view`/`transpose` by decompose.py rather than
    # listed here, because the exporter has no symbolic for either.
    torch.ops.aten.expand.default,
    # misc
    torch.ops.aten.softmax.int,
    # `_safe_softmax` is expanded by decompose.py into these public aten ops so
    # fully masked attention rows remain zero. The resulting ONNX graph was
    # compiled end-to-end through hbdk4 on S600; bool is restricted below to
    # exactly the mask-producing nodes rather than enabled for arbitrary ops.
    torch.ops.aten.eq.Scalar,
    torch.ops.aten.all.dim,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.where.self,
    # `_softmax` stays out: it has no ONNX symbolic. decompose.py rewrites it
    # to softmax.int before this runs.
    torch.ops.aten.clone.default,
    # Public dtype cast used by the BF16 artifact-promotion pass and fixed mask
    # construction. `_to_copy` is normalized to this in decompose.py.
    torch.ops.aten.to.dtype,
}

# Structural nodes that never block a partition: they carry no computation.
_STRUCTURAL = {"placeholder", "output", "get_attr"}

# Shape-only ops: they rearrange or broadcast a tensor without doing arithmetic
# on its values. A partition made of nothing but these has no MAC work to
# offload, and constant folding collapses it to nothing at all -- hbdk4 then
# rejects the artifact with "any func block must contain at least one op other
# than return op". Kept in _SUPPORTED so they ride along inside a real
# partition; only an all-shape region is dropped.
_SHAPE_ONLY = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
}


@dataclass
class Partition:
    """A maximal connected region of BPU-supported nodes."""

    nodes: list[Node] = field(default_factory=list)
    inputs: list[Node] = field(default_factory=list)
    outputs: list[Node] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)


_BF16_PROMOTABLE = {
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.cat.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.neg.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.silu.default,
    torch.ops.aten.softmax.int,
    torch.ops.aten.zeros_like.default,
    torch.ops.aten.where.self,
    torch.ops.aten.to.dtype,
}


def _tensor_values(node: Node) -> list[torch.Tensor]:
    val = node.meta.get("val")
    vals = val if isinstance(val, (tuple, list)) else [val]
    return [v for v in vals if isinstance(v, torch.Tensor)]


def is_supported(node: Node) -> bool:
    """Whether a single node can run on the BPU.

    Dynamic shapes are rejected: hbdk4 bakes memspace offsets and SRAM tiling
    into the compiled artifact, so every shape must be static at compile time.
    """
    if node.op in _STRUCTURAL:
        return False
    if node.op == "call_method":
        return False
    if node.op == "call_module":
        # Dynamo traces down to aten ops; a surviving call_module is opaque.
        return False
    if node.op != "call_function":
        return False
    if node.target in (operator.getitem,):
        # Tuple indexing is bookkeeping, absorbed into whichever side needs it.
        return True
    if node.target not in _SUPPORTED:
        return False

    val = node.meta.get("val")
    if val is None:
        return False
    vals = _tensor_values(node)
    for v in vals:
        if any(not isinstance(d, int) for d in v.shape):
            return False  # symbolic dim
        if v.dtype == torch.bfloat16 and node.target in _BF16_PROMOTABLE:
            continue
        if v.dtype not in (torch.float32, torch.float16, torch.int8, torch.int32):
            if not (
                v.dtype == torch.bool
                and node.target in (torch.ops.aten.eq.Scalar, torch.ops.aten.all.dim)
            ):
                return False
    return True


def partition_graph(
    gm: GraphModule, min_nodes: int = 3, min_compute_macs: int = 0
) -> list[Partition]:
    """Find maximal connected regions of supported nodes, in topological order.

    Connectivity, rather than adjacency in the graph's textual order, defines a
    region. An unsupported side branch may be emitted between two nodes on a
    supported data path (attention mask and rotary construction do exactly
    this); it becomes a boundary input but must not close the surrounding
    compute region. Conversely, an unsupported op *on* the data path leaves no
    supported edge across it and therefore still splits the graph.

    Regions smaller than `min_nodes` are dropped because their boundary cost is
    unlikely to be recovered. When `min_compute_macs > 0`, partitions whose
    estimated MAC count falls below that threshold are also rejected: BPU
    submission overhead is roughly 0.8 ms, so a partition must do enough work to
    amortize that cost.
    """
    supported = {n: is_supported(n) for n in gm.graph.nodes}
    supported_nodes = [n for n in gm.graph.nodes if supported[n]]
    parent = {n: n for n in supported_nodes}

    def find(node: Node) -> Node:
        root = node
        while parent[root] is not root:
            root = parent[root]
        while parent[node] is not node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left: Node, right: Node) -> None:
        left_root, right_root = find(left), find(right)
        if left_root is not right_root:
            parent[right_root] = left_root

    # Direct producer/consumer edges are the only evidence that two supported
    # nodes belong in the same executable region. Unsupported producers remain
    # ordinary partition inputs and do not connect otherwise independent work.
    for node in supported_nodes:
        for arg in node.all_input_nodes:
            if supported.get(arg, False):
                union(arg, node)

    groups: dict[Node, list[Node]] = {}
    for node in supported_nodes:
        groups.setdefault(find(node), []).append(node)

    position = {node: i for i, node in enumerate(gm.graph.nodes)}
    partitions = [
        Partition(nodes=nodes)
        for nodes in sorted(groups.values(), key=lambda ns: position[ns[0]])
        if any(n.target is not operator.getitem for n in nodes)
    ]
    for p in partitions:
        _compute_boundary(p)

    return [
        p
        for p in partitions
        if len(p) >= min_nodes
        and any(n.target not in _SHAPE_ONLY for n in p.nodes)
        and (min_compute_macs == 0 or _estimate_macs(p) >= min_compute_macs)
    ]


# Ops that perform meaningful arithmetic work: matrix multiply, convolution, and
# multi-head attention. Used to estimate whether a partition does enough MACs to
# recover BPU submission overhead (roughly 0.8 ms per call).
_COMPUTE_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}


def _estimate_macs(p: Partition) -> int:
    """Conservative MAC estimate from static shapes in node metadata.

    Returns 0 for partitions with no matrix/conv work, or when shapes are
    unavailable. A real hardware measurement would be better, but this is enough
    to reject tiny elementwise-only regions before compilation.
    """
    total = 0
    for node in p.nodes:
        if node.target not in _COMPUTE_OPS:
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        shape = val.shape
        if any(not isinstance(d, int) for d in shape):
            continue

        # mm(M, K) @ (K, N) -> M*K*N MACs
        if node.target in (
            torch.ops.aten.mm.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.addmm.default,
        ):
            args = node.all_input_nodes
            if len(args) < 2:
                continue
            lhs = args[0].meta.get("val")
            rhs = (
                args[1].meta.get("val")
                if node.target == torch.ops.aten.mm.default
                else args[0].meta.get("val")
            )
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                continue
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            if (
                len(lhs_shape) >= 2
                and len(rhs_shape) >= 2
                and all(isinstance(d, int) for d in lhs_shape)
                and all(isinstance(d, int) for d in rhs_shape)
            ):
                M = lhs_shape[-2] if len(lhs_shape) > 1 else 1
                K = lhs_shape[-1]
                N = rhs_shape[-1]
                total += M * K * N

        # bmm(B, M, K) @ (B, K, N) -> B*M*K*N MACs
        elif node.target is torch.ops.aten.bmm.default:
            args = node.all_input_nodes
            if len(args) < 2:
                continue
            lhs = args[0].meta.get("val")
            rhs = args[1].meta.get("val")
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                continue
            lhs_shape = lhs.shape
            rhs_shape = rhs.shape
            if (
                len(lhs_shape) == 3
                and len(rhs_shape) == 3
                and all(isinstance(d, int) for d in lhs_shape)
                and all(isinstance(d, int) for d in rhs_shape)
            ):
                B, M, K = lhs_shape
                N = rhs_shape[-1]
                total += B * M * K * N

        # conv2d: rough estimate (ignores padding/stride details)
        elif node.target in (
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv2d.default,
        ):
            # Output shape gives us the spatial extent; kernel and channels come
            # from weight. This is approximate.
            if len(shape) == 4:
                N, C_out, H_out, W_out = shape
                args = node.all_input_nodes
                if len(args) >= 2:
                    weight = args[1].meta.get("val")
                    if isinstance(weight, torch.Tensor) and len(weight.shape) == 4:
                        C_in, _, Kh, Kw = weight.shape
                        total += N * C_out * H_out * W_out * C_in * Kh * Kw

    return total


def _compute_boundary(p: Partition) -> None:
    """Fill in the tensors crossing into and out of a partition."""
    inside = set(p.nodes)

    seen_in: set[Node] = set()
    for node in p.nodes:
        for arg in node.all_input_nodes:
            if arg not in inside and arg not in seen_in:
                seen_in.add(arg)
                p.inputs.append(arg)

    seen_out: set[Node] = set()
    for node in p.nodes:
        # A node is an output if anything outside the partition consumes it.
        if any(user not in inside for user in node.users):
            if node not in seen_out:
                seen_out.add(node)
                p.outputs.append(node)
    p.outputs = [n for n in p.outputs if n.target is not operator.getitem] or p.outputs


def runtime_inputs(
    p: Partition, frozen: dict[str, torch.Tensor] | None = None
) -> list[Node]:
    """The partition inputs that must be passed at call time.

    Weights baked into the compiled artifact are excluded — see
    `extract_subgraph`. Zero-sized tensors are also excluded, as they cannot
    be represented in ONNX.
    """
    if not frozen:
        frozen = {}
    return [
        n
        for n in p.inputs
        if n.name not in frozen
        and not (
            isinstance(n.meta.get("val"), torch.Tensor)
            and n.meta["val"].numel() == 0
        )
    ]


def needs_bf16_promotion(p: Partition) -> bool:
    """Whether a region's floating-point contract contains BF16 tensors."""
    boundary = [*p.inputs, *p.outputs]
    return any(v.dtype == torch.bfloat16 for n in boundary for v in _tensor_values(n))


def extract_subgraph(
    gm: GraphModule,
    p: Partition,
    frozen: dict[str, torch.Tensor] | None = None,
    promote_bf16: bool = False,
) -> GraphModule:
    """Build a standalone GraphModule for one partition.

    The result takes the partition's boundary inputs as placeholders and
    returns its boundary outputs, so it can be exported to ONNX on its own.

    Boundary inputs named in `frozen` become module attributes instead of
    placeholders. AOTAutograd lifts every parameter and buffer to a graph
    input, so without this a 2-conv block crosses the boundary with 13 tensors
    instead of 1: each one is copied to the device on every call, and the ONNX
    exporter sees weights as graph inputs rather than initializers, which stops
    the QDQ pass from folding them to int8. Both effects are large enough to
    make the offloaded graph slower than eager.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    frozen = frozen or {}
    # BF16 promotion of frozen weights must happen outside FakeTensorMode
    if promote_bf16:
        with unset_fake_temporarily():
            frozen = {
                name: tensor.float() if tensor.dtype == torch.bfloat16 else tensor
                for name, tensor in frozen.items()
            }
    new_graph = Graph()
    env: dict[Node, Node] = {}
    consts: dict[str, torch.Tensor] = {}

    # Placeholders first, so the signature matches runtime_inputs() order.
    for inp in p.inputs:
        if inp.name in frozen:
            continue
        val = inp.meta.get("val")
        # Zero-sized tensors are handled by the decompose pass as constants
        if isinstance(val, torch.Tensor) and val.numel() == 0:
            # Map to a constant empty tensor in the subgraph
            with unset_fake_temporarily():
                const_name = f"_empty_{inp.name}"
                consts[const_name] = torch.empty(tuple(val.shape), dtype=val.dtype)
                node = new_graph.get_attr(const_name)
                node.meta = dict(inp.meta)
                env[inp] = node
            continue
        ph = new_graph.placeholder(inp.name)
        ph.meta = dict(inp.meta)
        if promote_bf16 and inp.meta.get("val") is not None:
            val = inp.meta["val"]
            if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
                ph.meta["val"] = val.float()
        env[inp] = ph

    for inp in p.inputs:
        if inp.name not in frozen:
            continue
        attr = f"_frozen_{inp.name}"
        consts[attr] = frozen[inp.name]
        node = new_graph.get_attr(attr)
        node.meta = dict(inp.meta)
        env[inp] = node

    for node in p.nodes:
        copied = new_graph.node_copy(node, lambda n: env[n])
        if promote_bf16:
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
                copied.meta = dict(copied.meta)
                copied.meta["val"] = val.float()
            if (
                copied.target is torch.ops.aten.to.dtype
                and copied.args[1] == torch.bfloat16
            ):
                copied.args = (copied.args[0], torch.float32, *copied.args[2:])
        env[node] = copied

    outputs = []
    for output in p.outputs:
        copied = env[output]
        val = output.meta.get("val")
        if (
            promote_bf16
            and isinstance(val, torch.Tensor)
            and val.dtype == torch.bfloat16
        ):
            # The copied graph executes entirely in F32. Keep its artifact output
            # in F32; `_BPUCall` restores BF16 at the outer FX boundary.
            copied.meta = dict(copied.meta)
            copied.meta["val"] = val.float()
        outputs.append(copied)

    new_graph.output(tuple(outputs))
    new_graph.lint()

    # GraphModule resolves every get_attr target against the root at
    # construction time, so the constants have to be on `gm` before the call,
    # not registered on the result afterwards. They are removed again below to
    # leave the caller's module as it was.
    # `detach` runs under the backend's active FakeTensorMode, which rejects
    # real tensors, so it has to be suspended (as with the ONNX export).
    with unset_fake_temporarily():
        for attr, t in consts.items():
            setattr(gm, attr, t.detach())
        try:
            sub = GraphModule(gm, new_graph)
        finally:
            for attr in consts:
                if hasattr(gm, attr):
                    delattr(gm, attr)

    sub.graph.eliminate_dead_code()
    sub.recompile()
    return sub


def summarize(gm: GraphModule, partitions: list[Partition]) -> str:
    """Human-readable partition report, for logs and debugging."""
    total = sum(1 for n in gm.graph.nodes if n.op == "call_function")
    offloaded = sum(1 for p in partitions for n in p.nodes if n.op == "call_function")
    lines = [
        f"{len(partitions)} BPU partition(s), "
        f"{offloaded}/{total} compute nodes offloaded"
    ]
    for i, p in enumerate(partitions):
        ops = [
            str(n.target).split(".")[-2] if "." in str(n.target) else str(n.target)
            for n in p.nodes
            if n.op == "call_function"
        ]
        macs = _estimate_macs(p)
        macs_str = f", ~{macs // 1_000_000}M MACs" if macs > 0 else ""
        lines.append(
            f"  [{i}] {len(p)} nodes, "
            f"{len(p.inputs)} in / {len(p.outputs)} out{macs_str}: {' '.join(ops[:8])}"
            + (" ..." if len(ops) > 8 else "")
        )
    return "\n".join(lines)
