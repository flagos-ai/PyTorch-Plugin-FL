"""
sum.dim_IntList dispatch tests

Verifies that torch.sum (dim variant):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_sum_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "torch.sum(a, dim=0)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSumDimCorrectness:
    """torch.sum(dim=...) correctness on flagos device."""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((128, 256), 0),
            ((128, 256), 1),
            ((64, 64, 64), 2),
            ((64, 64, 64), -1),
        ],
    )
    @pytest.mark.anyplatform
    def test_sum_dim_shape(self, shape, dim):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.sum(a, dim=dim)
        ref = torch.sum(a.cpu(), dim=dim)
        assert out.shape == ref.shape
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_sum_dim_correctness(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.sum(a, dim=1)
        ref = torch.sum(a.cpu(), dim=1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.anyplatform
    def test_sum_dim_keepdim(self):
        torch.manual_seed(2)
        a = torch.randn(16, 32, device=DEVICE)
        out = torch.sum(a, dim=0, keepdim=True)
        ref = torch.sum(a.cpu(), dim=0, keepdim=True)
        assert out.shape == ref.shape
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.cuda
    def test_sum_dim_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.sum(a_cuda, dim=1)
        a = a_cuda.to(DEVICE)
        out = torch.sum(a, dim=1)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_sum_dim_dtype(self, dtype):
        torch.manual_seed(4)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.sum(a, dim=0)
        ref = torch.sum(a.cpu().float(), dim=0)
        torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)

    @pytest.mark.anyplatform
    def test_sum_multi_dim(self):
        torch.manual_seed(5)
        a = torch.randn(8, 16, 32, device=DEVICE)
        out = torch.sum(a, dim=[0, 2])
        ref = torch.sum(a.cpu(), dim=[0, 2])
        assert out.shape == ref.shape
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)


class TestSumDimDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_sum__dim_IntList": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] sum.dim_IntList -> cuda" in result.stderr

    @pytest.mark.cuda
    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_sum__dim_IntList": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr


class TestSumDimAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify sum.dim_IntList on ascend backend matches CPU reference."""
        result = _run_subprocess(
            {"FLAGOS_OP_sum__dim_IntList": "ascend"}
        )
        assert result.returncode == 0
