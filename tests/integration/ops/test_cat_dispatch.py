"""
cat dispatch tests

Verifies that torch.cat:
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_cat_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference cat results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(32, 64, device="cuda:0", dtype=torch.float32)
    b = torch.randn(16, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.cat([a, b], dim=0)


def _run_cat_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a minimal cat call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch; "
        "a = torch.randn(4,4,device='flagos:0'); "
        "b = torch.randn(4,4,device='flagos:0'); "
        "torch.cat([a, b], dim=0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if check:
        assert result.returncode == 0, (
            f"Subprocess failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result


class TestCatDispatch:
    """torch.cat correctness on flagos device."""

    def test_cat_dim0(self):
        torch.manual_seed(0)
        a = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        b = torch.randn(16, 64, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=0)
        assert out.shape == (48, 64)
        assert out.device.type == "flagos"

    def test_cat_dim1(self):
        torch.manual_seed(1)
        a = torch.randn(32, 64, device=DEVICE, dtype=torch.float32)
        b = torch.randn(32, 128, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=1)
        assert out.shape == (32, 192)

    def test_cat_single_tensor(self):
        torch.manual_seed(2)
        a = torch.randn(8, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a], dim=0)
        assert out.shape == a.shape
        torch.testing.assert_close(out.cpu(), a.cpu())

    def test_cat_multiple_tensors(self):
        torch.manual_seed(3)
        tensors = [
            torch.randn(4, 8, device=DEVICE, dtype=torch.float32) for _ in range(5)
        ]
        out = torch.cat(tensors, dim=0)
        assert out.shape == (20, 8)

    def test_cat_negative_dim(self):
        torch.manual_seed(4)
        a = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        b = torch.randn(4, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=-1)
        assert out.shape == (4, 24)

    def test_cat_matches_cuda_ref(self, cuda_ref):
        """flagos cat result must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.cat([a, b], dim=0)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-5,
            atol=1e-5,
            msg="cat result on flagos differs from CUDA reference",
        )

    def test_cat_half(self):
        torch.manual_seed(5)
        a = torch.randn(16, 32, device=DEVICE, dtype=torch.float16)
        b = torch.randn(8, 32, device=DEVICE, dtype=torch.float16)
        out = torch.cat([a, b], dim=0)
        assert out.dtype == torch.float16
        assert out.shape == (24, 32)

    def test_cat_3d(self):
        torch.manual_seed(6)
        a = torch.randn(2, 4, 8, device=DEVICE, dtype=torch.float32)
        b = torch.randn(2, 4, 16, device=DEVICE, dtype=torch.float32)
        out = torch.cat([a, b], dim=2)
        assert out.shape == (2, 4, 24)


class TestCatDispatchLog:
    """Verify C++ wrapper routes to the correct backend."""

    def test_dispatch_log_flagos_default(self):
        """Default config routes cat to flagos."""
        result = _run_cat_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_cat": "flaggems"}
        )
        assert "[flagos dispatch] cat -> flagos" in result.stderr, (
            f"Expected flagos dispatch log, got:\n{result.stderr}"
        )

    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_cat=cuda overrides to cuda backend."""
        result = _run_cat_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_cat": "cuda"},
        )
        assert "[flagos dispatch] cat -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )
