"""
all dispatch tests

Verifies that torch.all:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_all_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_all_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.ones(4,4,device='flagos:0', dtype=torch.bool); "
        "torch.all(a)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestAllCorrectness:
    """torch.all correctness on flagos device."""

    def test_all_true(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True
        assert out.dtype == torch.bool

    def test_all_false(self):
        a = torch.zeros(32, 32, device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is False

    def test_all_mixed(self):
        a = torch.ones(32, 32, device=DEVICE, dtype=torch.bool)
        a[0, 0] = False
        out = torch.all(a)
        assert out.item() is False

    def test_all_empty_tensor(self):
        """Empty tensor: all() is vacuously true."""
        a = torch.tensor([], device=DEVICE, dtype=torch.bool)
        out = torch.all(a)
        assert out.item() is True

    def test_all_numeric_types(self):
        """all() should work on numeric types (non-zero = True)."""
        torch.manual_seed(0)
        a = torch.randn(16, 16, device=DEVICE)
        out = torch.all(a)
        ref = torch.all(a.cpu())
        assert out.item() == ref.item()

    def test_all_matches_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(1)
        a_cuda = torch.randint(0, 2, (64, 64), device="cuda:0", dtype=torch.bool)
        ref = torch.all(a_cuda)
        a = a_cuda.to(DEVICE)
        out = torch.all(a)
        assert out.item() == ref.item()

    @pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.int32])
    def test_all_dtypes(self, dtype):
        if dtype == torch.bool:
            a = torch.ones(16, 16, device=DEVICE, dtype=dtype)
        elif dtype.is_floating_point:
            a = torch.randn(16, 16, device=DEVICE, dtype=dtype) + 1.0
        else:
            a = torch.randint(1, 10, (16, 16), device=DEVICE, dtype=dtype)
        out = torch.all(a)
        assert out.dtype == torch.bool


class TestAllDispatch:
    """Verify dispatch routing and flaggems backend rejection."""

    def test_dispatch_log_cuda(self):
        result = _run_all_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_all": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] all -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        """Selecting flaggems backend must fail — not implemented."""
        result = _run_all_subprocess(
            {"FLAGOS_OP_all": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
