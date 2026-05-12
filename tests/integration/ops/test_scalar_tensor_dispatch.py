"""
scalar_tensor dispatch tests

Verifies that torch.scalar_tensor:
  - produces correct results on flagos device
  - C++ wrapper routes to cuda backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_scalar_tensor_dispatch.py -v
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
        "torch.scalar_tensor(3.14, device='flagos:0')"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestScalarTensorCorrectness:
    """torch.scalar_tensor correctness on flagos device."""

    def test_float_value(self):
        out = torch.scalar_tensor(3.14, device=DEVICE)
        assert out.shape == ()
        assert out.device.type == "flagos"
        torch.testing.assert_close(out.cpu(), torch.tensor(3.14))

    def test_int_value(self):
        out = torch.scalar_tensor(42, device=DEVICE, dtype=torch.int64)
        assert out.dtype == torch.int64
        assert out.item() == 42

    def test_zero(self):
        out = torch.scalar_tensor(0.0, device=DEVICE)
        assert out.item() == 0.0


class TestScalarTensorDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_scalar_tensor": "cuda"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] scalar_tensor -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_scalar_tensor": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
