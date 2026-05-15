"""
cos dispatch tests

Verifies that torch.cos:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_cos_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_cos_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; a = torch.randn(4,4,device='flagos:0'); torch.cos(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestCosCorrectness:
    """torch.cos correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    def test_cos_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.cos(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    def test_cos_values(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        out = torch.cos(a)
        ref = torch.cos(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_cos_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.cos(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.cos(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_cos_dtypes(self, dtype):
        torch.manual_seed(3)
        a = torch.randn(16, 16, device=DEVICE, dtype=dtype)
        out = torch.cos(a)
        assert out.dtype == dtype


class TestCosDispatch:
    """Verify dispatch routing and flaggems backend rejection."""

    def test_dispatch_log_cuda(self):
        result = _run_cos_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_cos": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] cos -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        """Selecting flaggems backend must fail — not implemented."""
        result = _run_cos_subprocess(
            {"FLAGOS_OP_cos": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
