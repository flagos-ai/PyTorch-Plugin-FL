"""
rsqrt dispatch tests

Verifies that torch.rsqrt:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_rsqrt_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_rsqrt_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0').abs() + 1e-6; "
        "torch.rsqrt(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestRsqrtCorrectness:
    """torch.rsqrt correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    def test_rsqrt_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE).abs() + 1e-6
        out = torch.rsqrt(a)
        assert out.shape == shape
        assert out.device.type == "flagos"

    def test_rsqrt_values(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE).abs() + 1e-6
        out = torch.rsqrt(a)
        ref = 1.0 / torch.sqrt(a.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_rsqrt_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(2)
        a_cuda = torch.randn(64, 64, device="cuda:0").abs() + 1e-6
        ref = torch.rsqrt(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.rsqrt(a)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestRsqrtDispatch:
    """Verify dispatch routing and flaggems backend rejection."""

    def test_dispatch_log_cuda(self):
        result = _run_rsqrt_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_rsqrt": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] rsqrt -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        """Selecting flaggems backend must fail — not implemented."""
        result = _run_rsqrt_subprocess(
            {"FLAGOS_OP_rsqrt": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
