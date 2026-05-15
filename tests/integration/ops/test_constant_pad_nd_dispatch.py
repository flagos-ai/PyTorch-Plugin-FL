"""
constant_pad_nd dispatch tests

Verifies that torch.nn.functional.pad (constant mode):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_constant_pad_nd_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "a = torch.randn(2,3,4,device='flagos:0'); "
        "F.pad(a, (1,1), mode='constant', value=0.0)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestConstantPadNdCorrectness:
    """constant_pad_nd correctness on flagos device."""

    def test_pad_1d(self):
        torch.manual_seed(0)
        a = torch.randn(4, 8, device=DEVICE)
        out = F.pad(a, (2, 3), mode="constant", value=0.0)
        assert out.shape == (4, 13)
        assert out.device.type == "flagos"
        # padded regions should be zero
        assert torch.all(out.cpu()[:, :2] == 0.0)
        assert torch.all(out.cpu()[:, -3:] == 0.0)

    def test_pad_2d(self):
        torch.manual_seed(1)
        a = torch.randn(2, 3, 4, device=DEVICE)
        out = F.pad(a, (1, 1, 2, 2), mode="constant", value=-1.0)
        assert out.shape == (2, 7, 6)
        # corners should be fill value
        assert out.cpu()[0, 0, 0].item() == -1.0

    def test_pad_matches_cpu(self):
        torch.manual_seed(2)
        a_cpu = torch.randn(3, 5, 7)
        ref = F.pad(a_cpu, (1, 2, 0, 1), mode="constant", value=0.5)

        a_fl = a_cpu.to(DEVICE)
        out = F.pad(a_fl, (1, 2, 0, 1), mode="constant", value=0.5)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_pad_dtype(self, dtype):
        a = torch.randn(4, 4, device=DEVICE, dtype=dtype)
        out = F.pad(a, (1, 1), mode="constant", value=0.0)
        assert out.dtype == dtype
        assert out.shape == (4, 6)


class TestConstantPadNdDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_constant_pad_nd": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] constant_pad_nd -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_constant_pad_nd": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
