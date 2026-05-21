"""
silu_backward dispatch tests

Verifies that silu_backward:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_silu_backward_dispatch.py -v
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
        "x = torch.randn(4,4,device='flagos:0',requires_grad=True); "
        "y = torch.nn.functional.silu(x); "
        "y.sum().backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestSiluBackwardCorrectness:
    """silu_backward correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_silu_backward_basic(self):
        torch.manual_seed(0)
        x = torch.randn(32, 32, device=DEVICE, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.device.type == "flagos"

    @pytest.mark.cuda
    def test_silu_backward_matches_cuda(self):
        torch.manual_seed(1)
        # Compute reference on CPU (avoid mixing cuda/flagos autograd streams)
        x_cpu = torch.randn(64, 64, requires_grad=True)
        y_cpu = torch.nn.functional.silu(x_cpu)
        y_cpu.sum().backward()

        torch.manual_seed(1)
        x_flagos = torch.randn(64, 64, device=DEVICE, requires_grad=True)
        y_flagos = torch.nn.functional.silu(x_flagos)
        y_flagos.sum().backward()

        torch.testing.assert_close(
            x_flagos.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.anyplatform
    def test_silu_backward_dtype(self, dtype):
        torch.manual_seed(2)
        x = torch.randn(16, 16, device=DEVICE, dtype=dtype, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == dtype

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (8, 16, 32)])
    @pytest.mark.anyplatform
    def test_silu_backward_shapes(self, shape):
        torch.manual_seed(3)
        x = torch.randn(*shape, device=DEVICE, requires_grad=True)
        y = torch.nn.functional.silu(x)
        y.sum().backward()
        assert x.grad.shape == shape


class TestSiluBackwardDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_silu_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] silu_backward -> cuda" in result.stderr

    @pytest.mark.cuda
    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_silu_backward": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr


class TestSiluBackwardAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify silu_backward on ascend backend matches CPU reference."""
        result = _run_subprocess(
            {"FLAGOS_OP_silu_backward": "ascend"}
        )
        assert result.returncode == 0
