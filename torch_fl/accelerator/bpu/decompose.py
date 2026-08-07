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

"""Rewrite aten ops that the ONNX exporter cannot handle.

AOTAutograd emits functional aten variants that the TorchScript-based ONNX
exporter has no symbolic function for. Each rewrite here preserves semantics
exactly and only changes which overload the graph names.
"""

from __future__ import annotations

import logging
import operator

import torch
from torch.fx import GraphModule

log = logging.getLogger("torch_fl.bpu")

_BN_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default


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

        users = list(node.users)
        # Only the primary output may be used; save_mean/save_invstd are
        # training-only statistics and have no ONNX equivalent.
        consumed = {
            u.args[1]
            for u in users
            if u.op == "call_function" and u.target is operator.getitem
        }
        if any(
            u.op != "call_function" or u.target is not operator.getitem for u in users
        ) or not consumed <= {0}:
            log.debug("batch_norm %s: non-trivial users, left alone", node.name)
            continue

        inp, weight, bias, running_mean, running_var, momentum, eps = node.args

        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(
                torch.ops.aten.batch_norm.default,
                args=(
                    inp,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    False,
                    momentum,
                    eps,
                    True,
                ),
            )
        new.meta.update(node.meta)
        val = node.meta.get("val")
        if isinstance(val, (tuple, list)) and val:
            new.meta["val"] = val[0]

        # Every user is getitem(node, 0); point them straight at `new`.
        for u in users:
            u.replace_all_uses_with(new)
            gm.graph.erase_node(u)
        gm.graph.erase_node(node)
        changed += 1

    return changed


def decompose_for_onnx(gm: GraphModule) -> GraphModule:
    """Apply every rewrite needed before torch.onnx.export. Mutates in place."""
    n = _rewrite_batch_norm(gm)
    if n:
        log.debug("decompose: rewrote %d batch_norm node(s)", n)
        gm.graph.lint()
        gm.recompile()
    return gm
