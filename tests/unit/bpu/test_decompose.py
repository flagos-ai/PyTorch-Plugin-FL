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

"""The batch-norm rewrite must preserve semantics and be ONNX-exportable."""

from __future__ import annotations

import operator
from pathlib import Path

import pytest
import torch
from torch._dynamo.backends.common import aot_autograd

from torch_fl.accelerator.bpu.compiler import export_onnx
from torch_fl.accelerator.bpu.decompose import decompose_for_onnx

BN_NO_TRAINING = torch.ops.aten._native_batch_norm_legit_no_training.default


class ConvBN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


def _aten_graph(mod: torch.nn.Module, x: torch.Tensor):
    """Capture the post-AOTAutograd aten graph, as the backend sees it."""
    captured = {}

    def fw(gm, inputs):
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)
    return captured["gm"]


def test_rewrites_batch_norm() -> None:
    gm = _aten_graph(ConvBN().eval(), torch.randn(1, 3, 8, 8))
    before = [n for n in gm.graph.nodes if n.target is BN_NO_TRAINING]
    assert before, "expected the functional batch_norm in the aten graph"

    decompose_for_onnx(gm)

    assert not [n for n in gm.graph.nodes if n.target is BN_NO_TRAINING]
    assert [n for n in gm.graph.nodes if n.target is torch.ops.aten.batch_norm.default]
    # The getitem that unpacked the tuple should be gone too.
    assert not [
        n
        for n in gm.graph.nodes
        if n.op == "call_function"
        and n.target is operator.getitem
        and n.args
        and getattr(n.args[0], "target", None) is BN_NO_TRAINING
    ]


def test_rewrite_preserves_numerics() -> None:
    """The rewritten graph must still reproduce eager output exactly."""
    mod = ConvBN().eval()
    x = torch.randn(2, 3, 8, 8)

    captured = {}

    def fw(gm, inputs):
        decompose_for_onnx(gm)
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        ref = mod(x)
        got = torch.compile(mod, backend=aot_autograd(fw_compiler=fw))(x)

    assert "gm" in captured
    assert not [n for n in captured["gm"].graph.nodes if n.target is BN_NO_TRAINING]
    torch.testing.assert_close(got, ref)


def test_exports_to_onnx_after_rewrite(tmp_path: Path) -> None:
    """Without the rewrite this raises UnsupportedOperatorError."""
    pytest.importorskip("onnx")
    mod = ConvBN().eval()
    x = torch.randn(1, 3, 8, 8)
    gm = _aten_graph(mod, x)

    inputs = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        v = n.meta["val"]
        inputs.append(torch.zeros(tuple(v.shape), dtype=v.dtype))

    out = tmp_path / "m.onnx"
    ins, outs = export_onnx(gm, inputs, out)
    assert out.exists() and ins and outs

    import onnx

    proto = onnx.load(str(out))
    onnx.checker.check_model(proto)
    # hbdk4's adaptor needs shapes for intermediates, which export_onnx adds.
    assert len(proto.graph.value_info) > 0
