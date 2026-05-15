"""
le.Tensor dispatch tests

Verifies that torch.le (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_le_dispatch.py -v
"""

import os
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
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.le(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestLeCorrectness:
    """torch.le correctness on flagos device."""

    def test_basic(self):
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        b = torch.tensor([2.0, 2.0, 2.0, 2.0], device=DEVICE)
        out = torch.le(a, b)
        expected = torch.tensor([True, True, False, False])
        torch.testing.assert_close(out.cpu(), expected)

    def test_output_dtype_is_bool(self):
        a = torch.randn(8, 8, device=DEVICE)
        b = torch.randn(8, 8, device=DEVICE)
        out = torch.le(a, b)
        assert out.dtype == torch.bool

    def test_matches_cpu(self):
        torch.manual_seed(0)
        a = torch.randn(64, 64, device=DEVICE)
        b = torch.randn(64, 64, device=DEVICE)
        out = torch.le(a, b)
        ref = torch.le(a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref)

    def test_broadcast(self):
        torch.manual_seed(1)
        a = torch.randn(4, 8, device=DEVICE)
        b = torch.randn(8, device=DEVICE)
        out = torch.le(a, b)
        ref = torch.le(a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref)


class TestLeDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_le__Tensor": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] le.Tensor -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_le__Tensor": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
