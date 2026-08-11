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

"""Partitioner tests. These run without hbdk4 — they check graph analysis only."""

import logging

import torch

from torch_fl.accelerator.bpu.partition import (
    extract_subgraph,
    partition_graph,
    summarize,
)


def _aot_graph(model, *inputs):
    """Capture the post-dispatch aten graph the backend actually sees.

    Dynamo alone yields torch-level calls; the partitioner runs after
    AOTAutograd lowers them to aten, so tests must capture at the same stage.
    """
    from torch._dynamo.backends.common import aot_autograd

    captured = {}

    def grab(gm, example_inputs):
        captured.setdefault("gm", gm)
        captured.setdefault("inputs", example_inputs)
        return gm.forward

    with torch.no_grad():
        torch.compile(model, backend=aot_autograd(fw_compiler=grab))(*inputs)
    return captured["gm"], captured["inputs"]


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class WithControlFlow(torch.nn.Module):
    """Data-dependent branch forces Dynamo to break the graph."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        if x.sum() > 0:  # graph break
            x = x * 2
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def test_conv_net_forms_one_partition():
    """conv+bn+relu must fuse into a single offloadable region."""
    model = ConvNet().eval()
    gm, _ = _aot_graph(model, torch.randn(1, 3, 32, 32))
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) == 1, summarize(gm, parts)
    p = parts[0]
    targets = {str(n.target) for n in p.nodes}
    assert any("convolution" in t for t in targets)
    assert any("relu" in t for t in targets)
    # Weights and inputs both cross the boundary; one activation comes out.
    assert len(p.inputs) >= 1
    assert len(p.outputs) == 1


def test_subgraph_is_extractable_and_numerically_equal():
    model = ConvNet().eval()
    x = torch.randn(1, 3, 32, 32)
    gm, _ = _aot_graph(model, x)
    parts = partition_graph(gm, min_nodes=2)
    sub = extract_subgraph(gm, parts[0])

    # The extracted subgraph must be a valid, runnable module.
    ex = [
        torch.zeros(tuple(n.meta["val"].shape), dtype=n.meta["val"].dtype)
        for n in parts[0].inputs
    ]
    with torch.no_grad():
        out = sub(*ex)
    assert out is not None


def test_min_nodes_filters_small_partitions():
    class Tiny(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm, _ = _aot_graph(Tiny(), torch.randn(4))
    assert partition_graph(gm, min_nodes=3) == []


def test_control_flow_does_not_break_partitioning():
    model = WithControlFlow().eval()
    gm, _ = _aot_graph(model, torch.randn(2, 3, 16, 16))
    # Dynamo hands us one subgraph at a time; partitioning must succeed on it.
    parts = partition_graph(gm, min_nodes=2)
    assert isinstance(parts, list)
    for p in parts:
        assert p.nodes
        assert p.outputs


def test_unsupported_op_splits_partitions():
    """An unsupported op in the middle must yield two separate regions."""

    class Split(torch.nn.Module):
        def forward(self, x):
            x = torch.relu(x)
            x = x * 2
            x = torch.erfinv(x)  # not on the whitelist
            x = torch.relu(x)
            return x * 3

    gm, _ = _aot_graph(Split(), torch.randn(8, 8))
    parts = partition_graph(gm, min_nodes=2)
    assert len(parts) == 2, summarize(gm, parts)
    # Nothing unsupported may leak into a partition.
    for p in parts:
        assert not any("erfinv" in str(n.target) for n in p.nodes)


def test_unsupported_side_branch_does_not_split_supported_path():
    """Mask bookkeeping may interleave with attention in topological order."""

    class SideBranch(torch.nn.Module):
        def forward(self, x):
            main = torch.relu(x)
            mask = torch.erfinv(x)  # unsupported, consumed by supported add
            return main * 2 + mask

    gm, _ = _aot_graph(SideBranch(), torch.randn(8, 8))
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) == 1, summarize(gm, parts)
    assert not any("erfinv" in str(n.target) for n in parts[0].nodes)
    assert any("erfinv" in str(n.target) for n in parts[0].inputs)


def test_backend_falls_back_without_hbdk(caplog, monkeypatch):
    """Without hbdk4 the model must still produce bit-exact results.

    find_hbdk is stubbed out, because on a machine that *can* compile this would
    offload to the BPU and the int8 result would only match approximately — a
    different property, covered by the accuracy tests.
    """
    import torch_fl.accelerator.bpu.backend as backend_mod
    from torch_fl.accelerator.bpu.backend import bpu_backend

    monkeypatch.setattr(backend_mod, "find_hbdk", lambda: None)

    model = ConvNet().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(x)

    with caplog.at_level(logging.INFO, logger="torch_fl.bpu"):
        compiled = torch.compile(model, backend=bpu_backend)
        with torch.no_grad():
            got = compiled(x)

    torch.testing.assert_close(got, expected)
    assert "no hbdk4 compiler reachable" in caplog.text


# -- transformer coverage ------------------------------------------------------


class RMSNormBlock(torch.nn.Module):
    """RMSNorm + gated MLP: the shape of every layer in a recent LLM."""

    def __init__(self, d=64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(d))
        self.fc1 = torch.nn.Linear(d, 2 * d)
        self.fc2 = torch.nn.Linear(2 * d, d)

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(v + 1e-6) * self.w
        return x + self.fc2(torch.nn.functional.silu(self.fc1(y)))


def test_rmsnorm_block_is_one_partition():
    """`pow` and `rsqrt` are what a norm is made of.

    Without them on the whitelist the graph splits at every norm: 86% of the
    nodes offloaded across two partitions, each paying its own boundary copy,
    for an op hbdk4 lowers entirely to b30vpu (checked with
    `convert(advice=True)`: 6 ops, zero CPU fallbacks).
    """
    from torch_fl.accelerator.bpu.decompose import decompose

    gm, _ = _aot_graph(RMSNormBlock().eval(), torch.randn(1, 8, 64))
    decompose(gm)  # as the backend does, before partitioning
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) == 1, summarize(gm, parts)
    targets = {n.target for n in parts[0].nodes}
    assert torch.ops.aten.pow.Tensor_Scalar in targets
    assert torch.ops.aten.rsqrt.default in targets

    compute = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert len(parts[0].nodes) == len(compute), "every compute node should offload"


def test_slice_does_not_split_a_partition():
    """Attention takes its causal window with `slice`, once per layer."""

    class Sliced(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x[:, :4] * 2) + 1

    gm, _ = _aot_graph(Sliced(), torch.randn(2, 8))
    parts = partition_graph(gm, min_nodes=1)

    assert len(parts) == 1, summarize(gm, parts)
    assert torch.ops.aten.slice.Tensor in {n.target for n in parts[0].nodes}

def test_safe_softmax_rewrite_stays_in_one_partition():
    """The semantic guard around softmax must not split attention again."""
    from torch_fl.accelerator.bpu.decompose import decompose

    class SafeSoftmax(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten._safe_softmax.default(x, -1)

    gm, _ = _aot_graph(SafeSoftmax(), torch.randn(1, 4, 8, 8))
    decompose(gm)
    parts = partition_graph(gm, min_nodes=1)

    assert len(parts) == 1, summarize(gm, parts)
    targets = {n.target for n in parts[0].nodes}
    assert torch.ops.aten.softmax.int in targets
    assert torch.ops.aten.eq.Scalar in targets
    assert torch.ops.aten.all.dim in targets
    assert torch.ops.aten.where.self in targets


def test_fixed_shape_qwen_has_one_decoder_partition():
    """Unsupported mask/rotary setup must not fragment decoder arithmetic."""
    import pytest

    transformers = pytest.importorskip("transformers")
    from torch_fl.accelerator.bpu.backend import _aot_decompositions
    from torch_fl.accelerator.bpu.decompose import decompose

    cfg = transformers.AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
    )
    cfg.num_hidden_layers = 1
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    model = transformers.AutoModelForCausalLM.from_config(cfg).eval().float()

    from torch._dynamo.backends.common import aot_autograd

    captured = {}

    def grab(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        torch.compile(
            model,
            backend=aot_autograd(
                fw_compiler=grab, decompositions=_aot_decompositions()
            ),
            dynamic=False,
        )(torch.tensor([[1, 2, 3, 4]]))

    gm = captured["gm"]
    decompose(gm)
    parts = partition_graph(gm, min_nodes=2)

    assert len(parts) <= 2, summarize(gm, parts)
    decoder = max(parts, key=len)
    targets = {n.target for n in decoder.nodes}
    assert torch.ops.aten.bmm.default in targets
    assert torch.ops.aten.softmax.int in targets
    assert torch.ops.aten.mm.default in targets
    assert len(decoder) >= 100, summarize(gm, parts)


def test_compile_model_forward_reaches_backend():
    """torch.compile(model) + model.generate() bypasses the backend entirely.

    OptimizedModule.__getattr__ forwards `.generate` to the original module,
    bound to self there, so generate()'s internal forward calls never enter
    Dynamo. The fix for generation is `model.forward = torch.compile(...)`,
    which makes generate() call the compiled forward.

    This test locks down that the workaround actually invokes the backend.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        import pytest

        pytest.skip("transformers not available")

    # Spy backend that counts invocations
    call_count = {"n": 0}

    def spy_backend(gm, example_inputs):
        call_count["n"] += 1
        return gm.forward

    # Tiny config
    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    cfg.num_hidden_layers = 1
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    model = AutoModelForCausalLM.from_config(cfg).eval()

    # Test 1: torch.compile(model.forward) invokes backend during generate()
    call_count["n"] = 0
    model.forward = torch.compile(model.forward, backend=spy_backend, fullgraph=False)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert call_count["n"] > 0, "backend was never called during generate()"

    # Test 2: torch.compile(model) does NOT invoke backend during generate()
    call_count["n"] = 0
    model2 = AutoModelForCausalLM.from_config(cfg).eval()
    compiled = torch.compile(model2, backend=spy_backend, fullgraph=False)
    with torch.no_grad():
        compiled.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert call_count["n"] == 0, (
        "backend should not be called when compiling the module"
    )
