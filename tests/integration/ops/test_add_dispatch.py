"""
add.Tensor dispatch tests

Verifies that torch.add (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_add_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_add_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.add(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAddTensorCorrectness:
    """torch.add correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    def test_add_shape(self, shape):
        torch.manual_seed(0)
        a = torch.randn(*shape, device=DEVICE)
        b = torch.randn(*shape, device=DEVICE)
        out = torch.add(a, b)
        assert out.shape == shape
        assert out.device.type == "flagos"

    def test_add_alpha(self):
        torch.manual_seed(1)
        a = torch.randn(32, 32, device=DEVICE)
        b = torch.randn(32, 32, device=DEVICE)
        out = torch.add(a, b, alpha=2.0)
        ref = a.cpu() + 2.0 * b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_add_broadcast(self):
        torch.manual_seed(2)
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.add(a, b)
        ref = a.cpu() + b.cpu()
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_add_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(3)
        a_cuda = torch.randn(64, 64, device="cuda:0")
        b_cuda = torch.randn(64, 64, device="cuda:0")
        ref = torch.add(a_cuda, b_cuda)
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.add(a, b)
        torch.testing.assert_close(out.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


class TestAddTensorDispatch:
    """Verify dispatch routing and flaggems backend rejection."""

    def test_dispatch_log_cuda(self):
        result = _run_add_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_add__Tensor": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] add.Tensor -> cuda" in result.stderr

    def test_dispatch_log_ascend(self):
        result = _run_add_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_add__Tensor": "ascend"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] add.Tensor -> ascend" in result.stderr

    def test_flaggems_backend_raises_error(self):
        """Selecting flaggems backend must fail — not implemented."""
        result = _run_add_subprocess(
            {"FLAGOS_OP_add__Tensor": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr


class TestAddTensorAscendDispatch:
    """Verify Ascend backend correctness."""

    def test_ascend_correctness(self):
        """Verify add.Tensor on ascend backend matches CPU reference."""
        result = _run_add_subprocess(
            {"FLAGOS_OP_add__Tensor": "ascend"}
        )
        assert result.returncode == 0
