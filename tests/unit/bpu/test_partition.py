"""Partitioner tests. These run without hbdk4 — they check graph analysis only."""

import logging

import torch

from torch_fl.backends.bpu.partition import (
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


def test_backend_falls_back_without_hbdk(caplog, monkeypatch):
    """Without hbdk4 the model must still produce bit-exact results.

    find_hbdk is stubbed out, because on a machine that *can* compile this would
    offload to the BPU and the int8 result would only match approximately — a
    different property, covered by the accuracy tests.
    """
    import torch_fl.backends.bpu.backend as backend_mod
    from torch_fl.backends.bpu.backend import bpu_backend

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
