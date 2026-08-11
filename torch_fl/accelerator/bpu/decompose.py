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

"""Rewrite aten ops that the ONNX exporter or the partitioner cannot handle.

AOTAutograd emits functional aten variants that the TorchScript-based ONNX
exporter has no symbolic function for, and multi-output variants whose extra
results are training-only. Each rewrite here preserves semantics exactly and
only changes which overload the graph names.

The pass runs twice: once on the whole aten graph before partitioning, because
a multi-output op splits a partition in two and its tuple cannot cross a
boundary, and once on each extracted subgraph before export. It is idempotent.
"""

from __future__ import annotations

import logging
import operator

import torch
from torch.fx import GraphModule

log = logging.getLogger("torch_fl.bpu")

_BN_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default
_MAX_POOL_INDICES = torch.ops.aten.max_pool2d_with_indices.default
_SOFTMAX = torch.ops.aten._softmax.default
_SAFE_SOFTMAX = torch.ops.aten._safe_softmax.default
_UNSAFE_VIEW = torch.ops.aten._unsafe_view.default
_T = torch.ops.aten.t.default
_ALIAS = torch.ops.aten.alias.default
_MUL_SCALAR = torch.ops.aten.mul.Scalar
_TO_COPY = torch.ops.aten._to_copy.default


def _only_uses_output(node, index: int) -> bool:
    """Whether every user of `node` is `getitem(node, index)`."""
    users = list(node.users)
    if not users:
        return False
    return all(
        u.op == "call_function"
        and u.target is operator.getitem
        and len(u.args) > 1
        and u.args[1] == index
        for u in users
    )


def _replace_tuple_op(gm: GraphModule, node, target, args) -> None:
    """Swap a tuple-returning node for a single-output equivalent."""
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(target, args=args)
    new.meta.update(node.meta)
    val = node.meta.get("val")
    if isinstance(val, (tuple, list)) and val:
        new.meta["val"] = val[0]

    for u in list(node.users):
        u.replace_all_uses_with(new)
        gm.graph.erase_node(u)
    gm.graph.erase_node(node)


def _rewrite_batch_norm(gm: GraphModule) -> int:
    """`_native_batch_norm_legit_no_training` -> `aten.batch_norm`.

    The functional op returns (out, save_mean, save_invstd); in inference only
    the first element is ever consumed, and `aten.batch_norm` with
    training=False computes exactly that and does have an ONNX symbolic.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _BN_NO_TRAINING:
            continue

        # Only the primary output may be used; save_mean/save_invstd are
        # training-only statistics and have no ONNX equivalent.
        if not _only_uses_output(node, 0):
            log.debug("batch_norm %s: non-trivial users, left alone", node.name)
            continue

        inp, weight, bias, running_mean, running_var, momentum, eps = node.args
        _replace_tuple_op(
            gm,
            node,
            torch.ops.aten.batch_norm.default,
            (inp, weight, bias, running_mean, running_var, False, momentum, eps, True),
        )
        changed += 1

    return changed


def _rewrite_max_pool(gm: GraphModule) -> int:
    """`max_pool2d_with_indices` -> `aten.max_pool2d`.

    This is what Dynamo emits for `nn.MaxPool2d`, and it is the op that used to
    cut ResNet in half. Two things go wrong with the tuple form. The partitioner
    cannot include it (the BPU produces no indices tensor), so the graph splits
    at the stem pool; and the pooled result then crosses the next partition's
    boundary as a `getitem`, whose producer's `meta['val']` is a *tuple*, which
    made `_example_inputs_for` reject that partition entirely -- reporting
    "dynamic shapes" for a graph that has none.

    The indices output only exists for max_unpool and the backward pass, so in
    an inference graph it is dead. When it is genuinely used the node is left
    alone and the old behaviour stands.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _MAX_POOL_INDICES:
            continue
        if not _only_uses_output(node, 0):
            log.debug("max_pool2d %s: indices are used, left alone", node.name)
            continue
        _replace_tuple_op(gm, node, torch.ops.aten.max_pool2d.default, node.args)
        changed += 1
    return changed


def _replace_1to1(gm: GraphModule, node, target, args) -> None:
    """Swap a node for a single-output equivalent with the same value."""
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(target, args=args)
    new.meta.update(node.meta)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)


def _rewrite_softmax(gm: GraphModule) -> int:
    """`aten._softmax` -> `aten.softmax.int`.

    The TorchScript ONNX exporter has a symbolic for `softmax` but none for the
    private `_softmax`, which is what AOTAutograd actually emits -- so any
    attention block failed to export with "Exporting the operator
    'aten::_softmax' to ONNX opset version 17 is not supported".

    The two differ only in the third argument: `_softmax(self, dim,
    half_to_float)` upcasts a half input to float when the flag is set, which is
    a training-time autocast concern. `softmax.int(self, dim, dtype)` expresses
    the same thing as an explicit output dtype, so the flag maps to
    dtype=float32 and False maps to dtype=None.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _SOFTMAX:
            continue
        self_, dim, half_to_float = node.args
        dtype = torch.float32 if half_to_float else None
        _replace_1to1(gm, node, torch.ops.aten.softmax.int, (self_, dim, dtype))
        changed += 1
    return changed


def _rewrite_safe_softmax(gm: GraphModule) -> int:
    """Lower `_safe_softmax` without losing its all-`-inf` row semantics.

    CPU flash attention uses this private op so a fully masked query produces
    zeros instead of NaNs. The public softmax is exportable; an explicit mask
    restores the one semantic difference without using a broad decomposition
    table (which would introduce unsupported prims operators).
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _SAFE_SOFTMAX:
            continue
        self_, dim = node.args[:2]
        dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
        with gm.graph.inserting_before(node):
            out = gm.graph.call_function(
                torch.ops.aten.softmax.int, args=(self_, dim, dtype)
            )
            masked = gm.graph.call_function(
                torch.ops.aten.eq.Scalar, args=(self_, float("-inf"))
            )
            masked_rows = gm.graph.call_function(
                torch.ops.aten.all.dim, args=(masked, dim, True)
            )
            zeros = gm.graph.call_function(torch.ops.aten.zeros_like.default, args=(out,))
            new = gm.graph.call_function(
                torch.ops.aten.where.self, args=(masked_rows, zeros, out)
            )

        self_val = self_.meta.get("val")
        if isinstance(self_val, torch.Tensor):
            out.meta.update(node.meta)
            masked.meta["val"] = self_val == float("-inf")
            masked_rows.meta["val"] = torch.all(masked.meta["val"], dim, True)
            zeros.meta.update(node.meta)
        new.meta.update(node.meta)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        changed += 1
    return changed


def _rewrite_alias_and_scalar_mul(gm: GraphModule) -> int:
    """Remove value aliases and name scalar multiply as the tensor overload.

    `alias` has no numerical effect in an inference graph. `mul.Scalar` and
    `mul.Tensor` share ONNX Mul semantics, but only the latter is accepted by
    the partitioner/export path. Calling the tensor overload with a Python
    scalar preserves aten's dtype-promotion behavior.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is _ALIAS:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            changed += 1
        elif node.target is _MUL_SCALAR:
            _replace_1to1(gm, node, torch.ops.aten.mul.Tensor, node.args)
            changed += 1
    return changed


def _rewrite_cast(gm: GraphModule) -> int:
    """Name dtype-only `_to_copy` casts as the public `aten.to.dtype` op.

    AOTAutograd emits the private copy form for autocast boundaries. The generic
    decomposition lowers it to `prims.convert_element_type`, which neither the
    exporter nor hbdk4 accepts; the public aten overload has the same value and
    copy semantics and a stable ONNX Cast symbolic.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _TO_COPY:
            continue
        dtype = node.kwargs.get("dtype")
        if dtype is None or any(
            key in node.kwargs
            for key in ("layout", "device", "pin_memory", "memory_format")
        ):
            continue
        args = (
            node.args[0],
            dtype,
            node.kwargs.get("non_blocking", False),
            True,
            None,
        )
        _replace_1to1(gm, node, torch.ops.aten.to.dtype, args)
        changed += 1
    return changed


def _rewrite_view_and_transpose(gm: GraphModule) -> int:
    """Rewrite the shape ops AOTAutograd emits that the partitioner rejects.

    `_unsafe_view` is `view` without the alias bookkeeping, emitted after a matmul;
    `t` is `transpose(0, 1)` restricted to 2-D. Both are pure reshapes with
    identical semantics to ops the BPU supports, and leaving them unsupported
    shattered a transformer block into nine partitions -- each one too small to
    pay for its own boundary copies.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target is _UNSAFE_VIEW:
            _replace_1to1(gm, node, torch.ops.aten.view.default, node.args)
            changed += 1
        elif node.target is _T:
            val = node.meta.get("val")
            # t() is the identity below 2-D; only the 2-D case is a transpose.
            if isinstance(val, torch.Tensor) and val.dim() == 2:
                _replace_1to1(
                    gm, node, torch.ops.aten.transpose.int, (node.args[0], 0, 1)
                )
                changed += 1
    return changed


def _rewrite_empty_cat(gm: GraphModule) -> int:
    """Replace cat([empty, x], dim) with x to avoid ONNX export errors.

    When the first forward has no past_key_values, the KV cache inputs are
    zero-sized tensors. Concatenating them with current K/V fails ONNX export
    because ONNX's cat symbolic function rejects empty tensors.
    """
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not torch.ops.aten.cat.default:
            continue

        # cat takes a list of tensors as first argument
        if not node.args or not isinstance(node.args[0], (list, tuple)):
            continue

        tensors = node.args[0]
        # Check if any input is a zero-sized tensor
        non_empty = []
        for t in tensors:
            if not hasattr(t, "meta") or "val" not in t.meta:
                non_empty.append(t)
                continue
            val = t.meta["val"]
            if not isinstance(val, torch.Tensor) or val.numel() > 0:
                non_empty.append(t)

        # If all tensors are non-empty, or all are empty, leave it alone
        if len(non_empty) == len(tensors) or len(non_empty) == 0:
            continue

        # If only one non-empty tensor remains, replace cat with it directly
        if len(non_empty) == 1:
            node.replace_all_uses_with(non_empty[0])
            gm.graph.erase_node(node)
            changed += 1
        # Otherwise, cat only the non-empty tensors
        elif len(non_empty) > 1:
            with gm.graph.inserting_before(node):
                new_args = (non_empty, *node.args[1:])
                new = gm.graph.call_function(torch.ops.aten.cat.default, args=new_args)
            new.meta.update(node.meta)
            node.replace_all_uses_with(new)
            gm.graph.erase_node(node)
            changed += 1

    return changed


def decompose(gm: GraphModule) -> GraphModule:
    """Apply every rewrite. Mutates in place and is safe to call twice."""
    n = (
        _rewrite_batch_norm(gm)
        + _rewrite_max_pool(gm)
        + _rewrite_softmax(gm)
        + _rewrite_safe_softmax(gm)
        + _rewrite_alias_and_scalar_mul(gm)
        + _rewrite_cast(gm)
        + _rewrite_view_and_transpose(gm)
        + _rewrite_empty_cat(gm)
    )
    if n:
        log.debug("decompose: rewrote %d node(s)", n)
        gm.graph.lint()
        gm.recompile()
    return gm


# The pass used to run only just before torch.onnx.export, which is why it was
# named for it. It now also runs before partitioning; kept as an alias because
# it is the spelling the tests and any out-of-tree caller use.
decompose_for_onnx = decompose
