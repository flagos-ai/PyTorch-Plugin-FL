"""
mean.dim dispatch tests

Verifies that torch.mean(dim=...):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_mean_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_mean_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,8,device='flagos:0'); "
        "torch.mean(a, dim=-1)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestMeanDimCorrectness:
    """torch.mean(dim=...) correctness on flagos device."""

    @pytest.mark.parametrize(
        "shape,dim",
        [
            ((128, 256), -1),
            ((64, 64, 64), 1),
            ((32, 16), 0),
        ],
    )
    def test_mean_shape(self, shape, dim):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        out = torch.mean(a, dim=dim)
        expected_shape = list(shape)
        del expected_shape[dim if dim >= 0 else len(shape) + dim]
        assert list(out.shape) == expected_shape
        assert out.device.type == "flagos"

    def test_mean_keepdim(self):
        torch.manual_seed(1)
        a = torch.randn(32, 64, device=DEVICE)
        out = torch.mean(a, dim=-1, keepdim=True)
        assert out.shape == (32, 1)

    def test_mean_values(self):
        torch.manual_seed(2)
        a = torch.randn(16, 32, device=DEVICE)
        out = torch.mean(a, dim=-1)
        ref = a.cpu().mean(dim=-1)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_mean_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.randn(64, 128, device="cuda:0")
        ref = torch.mean(a_cuda, dim=-1)
        a = a_cuda.to(DEVICE)
        out = torch.mean(a, dim=-1)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestMeanDimDispatch:
    """Verify dispatch routing and flaggems backend rejection."""

    def test_dispatch_log_cuda(self):
        result = _run_mean_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mean__dim": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] mean.dim -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        """Selecting flaggems backend must fail — not implemented."""
        result = _run_mean_subprocess(
            {"FLAGOS_OP_mean__dim": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
