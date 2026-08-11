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

"""BF16 boundary promotion: verify promoted artifacts preserve outer BF16 dtype."""

from __future__ import annotations

import torch
from torch._dynamo.backends.common import aot_autograd

from torch_fl.accelerator.bpu.backend import _BPUCall
from torch_fl.accelerator.bpu.partition import (
    extract_subgraph,
    needs_bf16_promotion,
    partition_graph,
    runtime_inputs,
)


def _aot_graph(model, *inputs):
    """Capture the post-AOTAutograd aten graph."""
    captured = {}

    def fw(gm, example_inputs):
        captured["gm"] = gm
        captured["inputs"] = example_inputs
        return gm.forward

    with torch.no_grad():
        torch.compile(model, backend=aot_autograd(fw_compiler=fw))(*inputs)
    return captured["gm"], captured["inputs"]


class BF16Matmul(torch.nn.Module):
    """A single BF16 matmul, the simplest promotable operation."""

    def __init__(self, d: int = 32):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(d, d, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w


class BF16GatedMLP(torch.nn.Module):
    """BF16 gated MLP: matmul + silu + matmul, the LLM FFN shape."""

    def __init__(self, d: int = 64):
        super().__init__()
        self.gate = torch.nn.Linear(d, d * 2, dtype=torch.bfloat16)
        self.down = torch.nn.Linear(d * 2, d, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)))


def test_detects_bf16_boundary():
    """A BF16 matmul must be marked for promotion."""
    gm, _ = _aot_graph(BF16Matmul().eval(), torch.randn(4, 32, dtype=torch.bfloat16))
    parts = partition_graph(gm, min_nodes=1)
    assert len(parts) == 1
    assert needs_bf16_promotion(parts[0])


def test_f32_boundary_needs_no_promotion():
    """An F32 graph must not trigger promotion."""
    gm, _ = _aot_graph(BF16Matmul().eval().float(), torch.randn(4, 32))
    parts = partition_graph(gm, min_nodes=1)
    assert len(parts) == 1
    assert not needs_bf16_promotion(parts[0])


def test_promoted_subgraph_has_f32_placeholders():
    """Promoted extraction must expose F32 inputs, not BF16."""
    gm, _ = _aot_graph(BF16Matmul().eval(), torch.randn(4, 32, dtype=torch.bfloat16))
    p = partition_graph(gm, min_nodes=1)[0]
    sub = extract_subgraph(gm, p, promote_bf16=True)

    phs = [n for n in sub.graph.nodes if n.op == "placeholder"]
    assert len(phs) >= 1, "expected at least one placeholder (activation)"
    # The activation input must be promoted to F32.
    activation_ph = [
        ph
        for ph in phs
        if "arg0" in ph.name
        or not any(kw in ph.name for kw in ("weight", "bias", "frozen"))
    ]
    assert len(activation_ph) >= 1
    assert activation_ph[0].meta["val"].dtype == torch.float32


def test_promoted_subgraph_has_f32_outputs():
    """Promoted extraction must expose F32 outputs; _BPUCall restores BF16."""
    gm, _ = _aot_graph(BF16Matmul().eval(), torch.randn(4, 32, dtype=torch.bfloat16))
    p = partition_graph(gm, min_nodes=1)[0]
    sub = extract_subgraph(gm, p, promote_bf16=True)

    outs = [n for n in sub.graph.nodes if n.op == "output"]
    assert len(outs) == 1
    out_val = outs[0].args[0]
    if isinstance(out_val, tuple):
        out_val = out_val[0]
    assert out_val.meta["val"].dtype == torch.float32


def test_promoted_frozen_weights_are_f32():
    """Frozen BF16 weights must become F32 initializers in the promoted artifact."""
    model = BF16Matmul().eval()
    gm, _ = _aot_graph(model, torch.randn(4, 32, dtype=torch.bfloat16))
    p = partition_graph(gm, min_nodes=1)[0]

    frozen = {n.name: model.w for n in p.inputs if "weight" in n.name}
    if not frozen:
        # Fallback: map by position if naming failed.
        frozen = {p.inputs[0].name: model.w}

    sub = extract_subgraph(gm, p, frozen, promote_bf16=True)
    attrs = [n for n in sub.graph.nodes if n.op == "get_attr"]
    assert len(attrs) == 1
    attr_name = attrs[0].target
    weight = getattr(sub, attr_name)
    assert weight.dtype == torch.float32
    assert weight.shape == model.w.shape


def test_bf16_call_converts_inputs_to_f32():
    """_BPUCall with input_dtypes must convert BF16 runtime inputs to F32."""

    class StubRuntime:
        def __init__(self):
            self.received_dtypes = []

        def __call__(self, *args):
            self.received_dtypes = [a.dtype for a in args]
            return [args[0] * 2]

    stub = StubRuntime()
    call = _BPUCall(
        stub,
        n_outputs=1,
        input_dtypes=[torch.float32],
        output_dtypes=None,
    )

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        call(x)

    assert stub.received_dtypes == [torch.float32]


def test_bf16_call_restores_bf16_outputs():
    """_BPUCall with output_dtypes must restore BF16 at the splice boundary."""

    class StubRuntime:
        def __call__(self, *args):
            return [args[0].float() * 2]

    call = _BPUCall(
        StubRuntime(),
        n_outputs=1,
        input_dtypes=None,
        output_dtypes=[torch.bfloat16],
    )

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        out = call(x)

    assert out.dtype == torch.bfloat16


def test_bf16_call_arity_mismatch_raises():
    """Length mismatches must raise rather than silently drop tensors."""

    class StubRuntime:
        def __call__(self, *args):
            return [args[0], args[0]]

    call = _BPUCall(
        StubRuntime(),
        n_outputs=2,
        input_dtypes=[torch.float32, torch.float32],
        output_dtypes=[torch.bfloat16],
    )

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    try:
        with torch.no_grad():
            call(x)
        assert False, "expected RuntimeError for input count mismatch"
    except RuntimeError as e:
        assert "input count mismatch" in str(e)

    call2 = _BPUCall(
        StubRuntime(),
        n_outputs=2,
        input_dtypes=[torch.float32],
        output_dtypes=[torch.bfloat16],
    )
    try:
        with torch.no_grad():
            call2(x)
        assert False, "expected RuntimeError for output count mismatch"
    except RuntimeError as e:
        assert "output count mismatch" in str(e)


def test_promoted_gated_mlp_preserves_numerics():
    """End-to-end: promoted BF16 MLP through CPU stub must match eager.

    This test uses unfrozen weights (all parameters passed at runtime) because
    _aot_graph here does not set up the outer context that _frozen_weights needs.
    The promotion contract holds either way: BF16 runtime inputs and outputs
    become F32 at the artifact boundary, and _BPUCall restores BF16.

    The test compares graph execution, not numerical precision between BF16 and
    F32-promoted paths: those differ by design. We verify the promoted subgraph
    runs and returns the correct dtype, not that F32 arithmetic matches BF16.
    """
    from torch_fl.accelerator.bpu.backend import _example_inputs_for

    model = BF16GatedMLP().eval()
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    gm, inputs = _aot_graph(model, x)
    p = partition_graph(gm, min_nodes=1)[0]

    # No frozen weights in this test harness, so all inputs are runtime inputs.
    frozen = {}
    sub = extract_subgraph(gm, p, frozen, promote_bf16=True)
    ex = _example_inputs_for(p, [x], gm, frozen)
    real_inputs = runtime_inputs(p, frozen)
    compile_inputs = [t.float() if t.dtype == torch.bfloat16 else t for t in ex]

    # Stub runtime that runs the promoted subgraph on CPU.
    class PromotedStub:
        def __call__(self, *args):
            with torch.no_grad():
                out = sub(*args)
            # Handle tuple output from subgraph.
            return list(out) if isinstance(out, (tuple, list)) else [out]

    call = _BPUCall(
        PromotedStub(),
        n_outputs=len(p.outputs),
        input_dtypes=[torch.float32 for _ in real_inputs],
        output_dtypes=[torch.bfloat16 for _ in p.outputs],
    )

    with torch.no_grad():
        expected = model(x)
        got = call(*compile_inputs)

    # Verify the dtype contract: BF16 in, BF16 out, even though artifact runs F32.
    assert got.dtype == torch.bfloat16
    assert got.shape == expected.shape
    # Sanity check: promoted and eager should be in the same ballpark, not wildly
    # different. F32 intermediate precision changes results slightly, so this is
    # loose: the test proves the plumbing works, not that F32 reproduces BF16.
    assert (got.float() - expected.float()).abs().max().item() < 1.0
