"""
where.self dispatch tests

Verifies that torch.where (condition, self, other):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_where_dispatch.py -v
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
        "cond = torch.tensor([True, False, True], device='flagos:0'); "
        "a = torch.tensor([1.0, 2.0, 3.0], device='flagos:0'); "
        "b = torch.tensor([4.0, 5.0, 6.0], device='flagos:0'); "
        "torch.where(cond, a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestWhereCorrectness:
    """torch.where correctness on flagos device."""

    def test_basic(self):
        cond = torch.tensor([True, False, True, False], device=DEVICE)
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEVICE)
        b = torch.tensor([5.0, 6.0, 7.0, 8.0], device=DEVICE)
        out = torch.where(cond, a, b)
        expected = torch.tensor([1.0, 6.0, 3.0, 8.0])
        torch.testing.assert_close(out.cpu(), expected)

    def test_matches_cpu(self):
        torch.manual_seed(0)
        x = torch.randn(32, 32, device=DEVICE)
        y = torch.randn(32, 32, device=DEVICE)
        cond = x > 0
        out = torch.where(cond, x, y)
        ref = torch.where(cond.cpu(), x.cpu(), y.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    def test_broadcast(self):
        torch.manual_seed(1)
        cond = torch.tensor([[True], [False], [True]], device=DEVICE)
        a = torch.randn(3, 4, device=DEVICE)
        b = torch.randn(3, 4, device=DEVICE)
        out = torch.where(cond, a, b)
        ref = torch.where(cond.cpu(), a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-5)

    def test_output_device(self):
        cond = torch.tensor([True, False], device=DEVICE)
        a = torch.tensor([1.0, 2.0], device=DEVICE)
        b = torch.tensor([3.0, 4.0], device=DEVICE)
        out = torch.where(cond, a, b)
        assert out.device.type == "flagos"


class TestWhereDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_where__self": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] where.self -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_where__self": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
