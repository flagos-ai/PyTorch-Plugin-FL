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

"""Cost-aware partition selection: reject regions that cannot amortize overhead."""

import torch

from torch_fl.accelerator.bpu.partition import _estimate_macs, partition_graph


def _aot_graph(model, *inputs):
    """Capture the post-AOTAutograd aten graph."""
    from torch._dynamo.backends.common import aot_autograd

    captured = {}

    def fw(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    with torch.no_grad():
        torch.compile(model, backend=aot_autograd(fw_compiler=fw))(*inputs)
    return captured["gm"]


class TinyMatmul(torch.nn.Module):
    """A single tiny matmul: 8x8x8 = 512 MACs, below any reasonable threshold."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return x @ self.w


class LargeMatmul(torch.nn.Module):
    """A large matmul: 128x128x128 = 2M MACs, well above overhead."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(128, 128))

    def forward(self, x):
        return x @ self.w


class ElementwiseOnly(torch.nn.Module):
    """No matrix work, only cheap pointwise ops."""

    def forward(self, x):
        return torch.relu(x + 1) * 2


def test_estimates_matmul_macs():
    """A single matmul's MAC count must match M*K*N."""
    gm = _aot_graph(LargeMatmul().eval(), torch.randn(128, 128))
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=0)
    assert len(parts) == 1
    macs = _estimate_macs(parts[0])
    assert macs == 128 * 128 * 128


def test_estimates_bmm_macs():
    """Batched matmul must include the batch dimension."""

    class BMM(torch.nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    gm = _aot_graph(BMM(), torch.randn(4, 16, 32), torch.randn(4, 32, 64))
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=0)
    assert len(parts) == 1
    macs = _estimate_macs(parts[0])
    assert macs == 4 * 16 * 32 * 64


def test_rejects_tiny_matmul_below_threshold():
    """A matmul too small to recover submission overhead must be dropped."""
    gm = _aot_graph(TinyMatmul().eval(), torch.randn(8, 8))
    # Threshold: 100K MACs. The 8x8x8 matmul has 512 MACs, well below.
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=100_000)
    assert len(parts) == 0


def test_keeps_large_matmul_above_threshold():
    """A matmul that does enough work must be retained."""
    gm = _aot_graph(LargeMatmul().eval(), torch.randn(128, 128))
    # Threshold: 1M MACs. The 128^3 matmul has 2M MACs, above threshold.
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=1_000_000)
    assert len(parts) == 1


def test_elementwise_only_has_zero_macs():
    """A region with no matrix/conv work must estimate to zero."""
    gm = _aot_graph(ElementwiseOnly(), torch.randn(128, 128))
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=0)
    assert len(parts) == 1
    assert _estimate_macs(parts[0]) == 0


def test_zero_threshold_keeps_all_partitions():
    """When min_compute_macs=0, the filter is disabled entirely."""
    gm = _aot_graph(TinyMatmul().eval(), torch.randn(8, 8))
    parts = partition_graph(gm, min_nodes=1, min_compute_macs=0)
    assert len(parts) == 1


def test_min_nodes_still_applies():
    """MAC threshold does not replace the node-count filter."""

    class OneAdd(torch.nn.Module):
        def forward(self, x):
            return x + 1

    gm = _aot_graph(OneAdd(), torch.randn(8))
    # min_nodes=3, so this single-node partition is dropped regardless of MACs.
    parts = partition_graph(gm, min_nodes=3, min_compute_macs=0)
    assert len(parts) == 0
