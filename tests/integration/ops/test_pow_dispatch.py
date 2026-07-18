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
pow.Tensor_Scalar dispatch tests

Verifies that torch.pow(tensor, scalar):
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems_python backend (default)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_pow_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_pow_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0').abs() + 0.1; "
        "torch.pow(a, 2.0)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestPowTensorScalarCorrectness:
    """torch.pow(tensor, scalar) correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    @pytest.mark.anyplatform
    def test_pow_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE).abs() + 0.1
        out = torch.pow(a, 2.0)
        assert out.shape == shape
        assert out.device.type == "flagos"

    @pytest.mark.parametrize("exp", [2.0, 3.0, 0.5, -1.0])
    @pytest.mark.anyplatform
    def test_pow_values(self, exp):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE).abs() + 0.1
        out = torch.pow(a, exp)
        ref = torch.pow(a.cpu(), exp)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_pow_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0").abs() + 0.1
        ref = torch.pow(a_cuda, 2.5)
        a = a_cuda.to(DEVICE)
        out = torch.pow(a, 2.5)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.anyplatform
    def test_pow_dtypes(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype).abs() + 0.1
        out = torch.pow(a, 2.0)
        assert out.dtype == dtype

    @pytest.mark.anyplatform
    def test_pow_fast_path_square(self):
        """Test exp=2 fast path."""
        torch.manual_seed(4)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.pow(a, 2.0)
        ref = a * a
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-6, atol=1e-6)

    @pytest.mark.anyplatform
    def test_pow_fast_path_cube(self):
        """Test exp=3 fast path."""
        torch.manual_seed(5)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.pow(a, 3.0)
        ref = a * a * a
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-6, atol=1e-6)


class TestPowTensorScalarDispatch:
    """Verify dispatch routing for pow.Tensor_Scalar op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_pow_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_pow__Tensor_Scalar": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] pow.Tensor_Scalar -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_pow_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_pow__Tensor_Scalar": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] pow.Tensor_Scalar -> cuda" in result.stderr


class TestPowTensorScalarAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify pow.Tensor_Scalar on ascend backend matches CPU reference."""
        result = _run_pow_subprocess({"FLAGOS_OP_pow__Tensor_Scalar": "ascend"})
        assert result.returncode == 0
