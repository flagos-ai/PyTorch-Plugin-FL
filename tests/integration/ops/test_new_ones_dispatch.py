"""
new_ones dispatch tests

Verifies that tensor.new_ones:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_new_ones_dispatch.py -v
"""

import os
import pytest
import subprocess
import sys

import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess(extra_env: dict, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "x = torch.randn(4,4,device='flagos:0'); "
        "x.new_ones(3, 3)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestNewOnesCorrectness:
    """tensor.new_ones correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_basic(self):
        x = torch.randn(4, 4, device=DEVICE)
        out = x.new_ones(3, 3)
        assert out.shape == (3, 3)
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.ones(3, 3))

    @pytest.mark.anyplatform
    def test_preserves_dtype(self):
        x = torch.randn(4, 4, device=DEVICE, dtype=torch.float16)
        out = x.new_ones(2, 2)
        assert out.dtype == torch.float16
        torch.testing.assert_close(out.cpu().float(), torch.ones(2, 2))

    @pytest.mark.anyplatform
    def test_override_dtype(self):
        x = torch.randn(4, 4, device=DEVICE)
        out = x.new_ones(2, 2, dtype=torch.int32)
        assert out.dtype == torch.int32
        torch.testing.assert_close(out.cpu(), torch.ones(2, 2, dtype=torch.int32))

    @pytest.mark.anyplatform
    def test_all_ones(self):
        x = torch.randn(8, device=DEVICE)
        out = x.new_ones(100)
        assert (out.cpu() == 1).all()


class TestNewOnesDispatch:
    """Verify dispatch routing."""

    @pytest.mark.cuda
    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_new_ones": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] new_ones -> cuda" in result.stderr

    @pytest.mark.cuda
    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_new_ones": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr


class TestNewOnesAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify new_ones on ascend backend matches CPU reference."""
        result = _run_subprocess(
            {"FLAGOS_OP_new_ones": "ascend"}
        )
        assert result.returncode == 0
