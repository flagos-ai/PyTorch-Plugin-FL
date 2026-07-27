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

"""
narrow / diff dispatch tests

Regression coverage for the flagos ``narrow`` view kernel. The hand-written
kernel used to call ``self.narrow(...)`` (the Tensor member method), which
re-dispatches through PrivateUse1 back into the same kernel -> infinite
recursion -> stack overflow (SIGSEGV). It now calls ``at::native::narrow_symint``
directly (pure metadata, redispatches only to the registered ``slice`` kernel).

``torch.diff`` is a CompositeImplicitAutograd op that decomposes into ``narrow``,
so it segfaulted for the same reason; it is exercised here too because the
transformers causal-mask path (find_packed_sequence_indices) relies on it.

Usage:
    pytest tests/integration/ops/test_narrow_dispatch.py -v
"""

import pytest

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


class TestNarrowCorrectness:
    """narrow must not recurse and must match CPU semantics."""

    @pytest.mark.anyplatform
    def test_narrow_basic(self):
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.narrow(x, 1, 1, 7)
        assert tuple(y.shape) == (1, 7)
        torch.testing.assert_close(y.cpu(), torch.arange(1, 8).unsqueeze(0))

    @pytest.mark.anyplatform
    def test_narrow_negative_dim(self):
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.narrow(x, -1, 2, 4)
        torch.testing.assert_close(y.cpu(), torch.arange(2, 6).unsqueeze(0))

    @pytest.mark.anyplatform
    def test_narrow_full_length(self):
        # length == full size: exercises the no-op narrow that still re-entered
        # the kernel in the recursion bug.
        x = torch.arange(8, device=DEVICE).unsqueeze(0)
        y = torch.ops.aten.narrow.default(x, 1, 0, 8)
        torch.testing.assert_close(y.cpu(), x.cpu())

    @pytest.mark.anyplatform
    def test_narrow_matches_cpu_2d(self):
        torch.manual_seed(0)
        x_cpu = torch.randn(4, 16)
        x_fl = x_cpu.to(DEVICE)
        torch.testing.assert_close(
            torch.narrow(x_fl, 1, 3, 10).cpu(), torch.narrow(x_cpu, 1, 3, 10)
        )


class TestDiffCorrectness:
    """torch.diff decomposes into narrow; must not segfault."""

    @pytest.mark.anyplatform
    def test_diff_basic(self):
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        d = torch.diff(pos, dim=-1)
        assert tuple(d.shape) == (1, 31)
        torch.testing.assert_close(d.cpu(), torch.ones(1, 31, dtype=torch.long))

    @pytest.mark.anyplatform
    def test_diff_with_prepend(self):
        # The transformers find_packed_sequence_indices pattern.
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        first = pos[:, :1] - 1
        d = torch.diff(pos, prepend=first, dim=-1)
        d_cpu = torch.diff(pos.cpu(), prepend=first.cpu(), dim=-1)
        assert tuple(d.shape) == (1, 32)
        torch.testing.assert_close(d.cpu(), d_cpu)

    @pytest.mark.anyplatform
    def test_diff_then_cumsum(self):
        pos = torch.arange(32, device=DEVICE).unsqueeze(0)
        first = pos[:, :1] - 1
        mask = (torch.diff(pos, prepend=first, dim=-1) != 1).cumsum(-1)
        # single contiguous sequence -> all zeros
        assert (mask[:, -1] == 0).all().cpu().item()
