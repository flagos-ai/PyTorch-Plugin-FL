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
FlagGems RNG reproducibility + distribution tests

Regression coverage for the unified flagos RNG mechanism (commit 20f3876):
flaggems RNG now reads seed+offset from torch.cuda.default_generators[device]
(a per-device philox CUDA generator installed by the vendor compat shim), and
torch.cuda.manual_seed* is routed to those generators. This replaced the fragile
_patch_flaggems_philox() monkeypatch, which seeded once and ignored later
manual_seed calls.

The contract verified here:
  1. Reproducibility: same torch.manual_seed -> identical draws.
  2. Seed actually takes effect: different seeds -> different draws.
  3. Distributions are statistically correct (mean/variance within tolerance).

These only exercise the flaggems Triton RNG kernels, so they are marked
flaggems_python and run under FLAGOS_USE_FLAGGEMS=1. Without flaggems the same
ops route to the CUDA boxing path (native generator), which is covered
elsewhere.

Usage:
    FLAGOS_METAX_BOXING=1 FLAGOS_USE_FLAGGEMS=1 \
        pytest tests/integration/ops/test_rng_dispatch.py -v
"""

import pytest

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"

# Every test here is marked @pytest.mark.flaggems_python. The shared ops/conftest.py
# skips flaggems_python-marked tests unless FLAGOS_USE_FLAGGEMS is enabled, because
# these assert the RNG *reproducibility contract* which only holds when the runtime
# routes RNG through the FlagGems Triton generator (torch.cuda.default_generators
# seed/offset). In pure boxing mode the same ops fall back to the native CUDA
# generator, where torch.manual_seed does not take effect on the flagos device ->
# not reproducible. The gate lives in conftest so it applies uniformly.


def _reproducible(op):
    """Run op() twice under the same seed; return (is_reproducible, first_draw)."""
    torch.manual_seed(1234)
    a = op()
    torch.manual_seed(1234)
    b = op()
    return torch.equal(a.cpu(), b.cpu()), a


_RNG_OPS = {
    "rand": lambda: torch.rand(256, device=DEVICE),
    "randn": lambda: torch.randn(256, device=DEVICE),
    "uniform_": lambda: torch.empty(256, device=DEVICE).uniform_(),
    "exponential_": lambda: torch.empty(256, device=DEVICE).exponential_(),
    "multinomial": lambda: torch.multinomial(
        torch.ones(32, device=DEVICE), 16, replacement=True
    ),
}


class TestRngReproducible:
    @pytest.mark.flaggems_python
    @pytest.mark.parametrize("name", list(_RNG_OPS))
    def test_same_seed_same_draw(self, name):
        same, _ = _reproducible(_RNG_OPS[name])
        assert same, f"{name} not reproducible under torch.manual_seed"

    @pytest.mark.flaggems_python
    def test_different_seed_differs(self):
        # Guards against the old bug where the seed was fixed once and ignored.
        torch.manual_seed(1)
        a = torch.randn(1000, device=DEVICE)
        torch.manual_seed(2)
        b = torch.randn(1000, device=DEVICE)
        assert not torch.equal(a.cpu(), b.cpu())


class TestRngDistribution:
    @pytest.mark.flaggems_python
    def test_rand_uniform_0_1(self):
        torch.manual_seed(42)
        r = torch.rand(100_000, device=DEVICE).cpu()
        assert abs(r.mean().item() - 0.5) < 0.02
        assert r.min().item() >= 0.0 and r.max().item() <= 1.0

    @pytest.mark.flaggems_python
    def test_randn_standard_normal(self):
        torch.manual_seed(42)
        n = torch.randn(100_000, device=DEVICE).cpu()
        assert abs(n.mean().item()) < 0.02
        assert abs(n.std().item() - 1.0) < 0.02

    @pytest.mark.flaggems_python
    def test_exponential_rate_1(self):
        torch.manual_seed(42)
        e = torch.empty(100_000, device=DEVICE).exponential_().cpu()
        assert abs(e.mean().item() - 1.0) < 0.05  # Exp(1): mean 1
        assert e.min().item() >= 0.0

    @pytest.mark.flaggems_python
    def test_uniform_range(self):
        torch.manual_seed(42)
        u = torch.empty(100_000, device=DEVICE).uniform_(2.0, 5.0).cpu()
        assert abs(u.mean().item() - 3.5) < 0.05
        assert u.min().item() >= 2.0 and u.max().item() <= 5.0
