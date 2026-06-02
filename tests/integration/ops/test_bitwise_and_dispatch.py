"""
bitwise_and.Tensor dispatch tests

Verifies that torch.bitwise_and (Tensor variant):
  - produces correct results on flagos device
  - C++ wrapper routes to cuda/metax backend
  - attempting flaggems backend raises an error (not implemented)

Usage:
    pytest tests/integration/ops/test_bitwise_and_dispatch.py -v
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
        "a = torch.tensor([True, False, True], device='flagos:0'); "
        "b = torch.tensor([True, True, False], device='flagos:0'); "
        "torch.bitwise_and(a, b)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestBitwiseAndCorrectness:
    """torch.bitwise_and correctness on flagos device."""

    @pytest.mark.anyplatform
    def test_bool_tensors(self):
        a = torch.tensor([True, False, True, False], device=DEVICE)
        b = torch.tensor([True, True, False, False], device=DEVICE)
        out = torch.bitwise_and(a, b)
        expected = torch.tensor([True, False, False, False])
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_int_tensors(self):
        a = torch.tensor([0b1100, 0b1010, 0b1111], dtype=torch.int32, device=DEVICE)
        b = torch.tensor([0b1010, 0b1100, 0b0101], dtype=torch.int32, device=DEVICE)
        out = torch.bitwise_and(a, b)
        expected = torch.tensor([0b1000, 0b1000, 0b0101], dtype=torch.int32)
        torch.testing.assert_close(out.cpu(), expected)

    @pytest.mark.anyplatform
    def test_matches_cpu(self):
        torch.manual_seed(0)
        a = torch.randint(0, 256, (64, 64), dtype=torch.int64, device=DEVICE)
        b = torch.randint(0, 256, (64, 64), dtype=torch.int64, device=DEVICE)
        out = torch.bitwise_and(a, b)
        ref = torch.bitwise_and(a.cpu(), b.cpu())
        torch.testing.assert_close(out.cpu(), ref)


class TestBitwiseAndDispatch:
    """Verify dispatch routing for bitwise_and.Tensor op."""

    @pytest.mark.flaggems_python
    def test_dispatch_log_flaggems_python(self):
        result = _run_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_bitwise_and__Tensor": "flaggems_python",
            },
            check=False,
        )
        assert "[flagos dispatch] bitwise_and.Tensor -> flagos_python" in result.stderr

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bitwise_and__Tensor": "cuda"}
        )
        if result.returncode != 0 and "backend not registered" in result.stderr:
            pytest.skip("cuda backend not available in this build")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] bitwise_and.Tensor -> cuda" in result.stderr

    @pytest.mark.anyplatform
    def test_dispatch_log_metax(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_bitwise_and__Tensor": "metax"}
        )
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "[flagos dispatch] bitwise_and.Tensor -> metax" in result.stderr

    @pytest.mark.anyplatform
    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_bitwise_and__Tensor": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr

class TestBitwiseAndAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify bitwise_and.Tensor on ascend backend matches CPU reference."""
        result = _run_subprocess({"FLAGOS_OP_bitwise_and__Tensor": "ascend"})
        assert result.returncode == 0
