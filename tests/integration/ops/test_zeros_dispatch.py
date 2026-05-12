"""
zeros dispatch tests

Verifies that torch.zeros:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_zeros_dispatch.py -v
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
        "torch.zeros(4, 4, device='flagos:0')"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestZerosCorrectness:
    """torch.zeros correctness on flagos device."""

    @pytest.mark.parametrize("shape", [(128, 256), (1,), (64, 64, 64)])
    def test_zeros_shape(self, shape):
        out = torch.zeros(*shape, device=DEVICE)
        assert out.shape == shape
        assert out.device.type == "flagos"
        assert torch.all(out.cpu() == 0.0)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
    def test_zeros_dtype(self, dtype):
        out = torch.zeros(8, 8, device=DEVICE, dtype=dtype)
        assert out.dtype == dtype
        assert torch.all(out.cpu() == 0)

    def test_zeros_scalar(self):
        out = torch.zeros((), device=DEVICE)
        assert out.shape == ()
        assert out.cpu().item() == 0.0


class TestZerosDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_zeros": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] zeros -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_zeros": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
