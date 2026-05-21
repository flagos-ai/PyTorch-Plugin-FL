"""
mm dispatch tests

Verifies that torch.mm and torch.mm.out:
  - produce correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_mm_dispatch.py -v
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
    """Reference mm results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    a = torch.randn(128, 256, device="cuda:0", dtype=torch.float32)
    b = torch.randn(256, 64, device="cuda:0", dtype=torch.float32)
    return a, b, torch.mm(a, b)


def _run_mm_subprocess(
    extra_env: dict, use_out: bool = False, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a minimal mm call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    if use_out:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(4,4,device='flagos:0'); "
            "b = torch.randn(4,4,device='flagos:0'); "
            "out = torch.empty(4,4,device='flagos:0'); "
            "torch.mm(a, b, out=out)"
        )
    else:
        code = (
            "import torch_fl, torch; "
            "a = torch.randn(4,4,device='flagos:0'); "
            "b = torch.randn(4,4,device='flagos:0'); "
            "torch.mm(a, b)"
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


class TestMmDispatch:
    """torch.mm correctness and cross-device consistency."""

    @pytest.mark.parametrize("M,K,N", [(128, 256, 64), (1, 1, 1), (512, 512, 512)])
    def test_mm_shape(self, M, K, N):
        torch.manual_seed(0)
        a = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
        b = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
        out = torch.mm(a, b)
        assert out.shape == (M, N)
        assert out.device.type == "flagos"

    def test_mm_out(self):
        torch.manual_seed(1)
        a = torch.randn(64, 128, device=DEVICE, dtype=torch.float32)
        b = torch.randn(128, 32, device=DEVICE, dtype=torch.float32)
        out = torch.empty(64, 32, device=DEVICE, dtype=torch.float32)
        ret = torch.mm(a, b, out=out)
        assert ret.data_ptr() == out.data_ptr()
        assert out.shape == (64, 32)

    def test_mm_matches_cuda_ref(self, cuda_ref):
        """flagos mm result must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        a_cuda, b_cuda, ref = cuda_ref
        a = a_cuda.to(DEVICE)
        b = b_cuda.to(DEVICE)
        out = torch.mm(a, b)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-3,
            atol=1e-3,
            msg="mm result on flagos differs from CUDA reference",
        )

    def test_mm_non_contiguous(self):
        torch.manual_seed(2)
        a = torch.randn(64, 128, device=DEVICE).t()
        b = torch.randn(64, 32, device=DEVICE)
        out = torch.mm(a, b)
        assert out.shape == (128, 32)

    def test_mm_half(self):
        torch.manual_seed(3)
        a = torch.randn(64, 128, device=DEVICE, dtype=torch.float16)
        b = torch.randn(128, 32, device=DEVICE, dtype=torch.float16)
        out = torch.mm(a, b)
        assert out.dtype == torch.float16
        assert out.shape == (64, 32)


class TestMmDispatchLog:
    """Verify C++ wrapper routes to the correct backend."""

    def test_dispatch_log_flagos_flaggems_override(self):
        """FLAGOS_OP_mm=flaggems routes mm to flagos backend."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm": "flaggems"}
        )
        assert "[flagos dispatch] mm -> flagos" in result.stderr, (
            f"Expected flagos dispatch log, got:\n{result.stderr}"
        )

    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_mm=cuda overrides to cuda backend."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm": "cuda"}
        )
        assert "[flagos dispatch] mm -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )

<<<<<<< HEAD
    def test_dispatch_log_ascend_override(self):
        """FLAGOS_OP_mm=ascend overrides to ascend backend."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm": "ascend"}
        )
        assert "[flagos dispatch] mm -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )

=======
>>>>>>> main
    def test_dispatch_log_mm_out_flagos_flaggems_override(self):
        """FLAGOS_OP_mm__out=flaggems routes mm.out to flagos backend."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm__out": "flaggems"},
            use_out=True,
        )
        assert "[flagos dispatch] mm.out -> flagos" in result.stderr, (
            f"Expected flagos dispatch log, got:\n{result.stderr}"
        )

    def test_dispatch_log_mm_out_cuda_override(self):
        """FLAGOS_OP_mm__out=cuda overrides mm.out to cuda."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm__out": "cuda"},
            use_out=True,
        )
        assert "[flagos dispatch] mm.out -> cuda" in result.stderr, (
            f"Expected cuda dispatch log, got:\n{result.stderr}"
        )

    def test_dispatch_log_mm_out_ascend_override(self):
        """FLAGOS_OP_mm__out=ascend overrides mm.out to ascend."""
        result = _run_mm_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_mm__out": "ascend"},
            use_out=True,
        )
        assert "[flagos dispatch] mm.out -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )


class TestMmAscendDispatch:
    """Verify Ascend backend correctness."""

    def test_ascend_correctness(self):
        """Verify mm on ascend backend matches CPU reference."""
        result = _run_mm_subprocess(
            {"FLAGOS_OP_mm": "ascend"}
        )
        assert result.returncode == 0

    def test_ascend_out_correctness(self):
        """Verify mm.out on ascend backend matches CPU reference."""
        result = _run_mm_subprocess(
            {"FLAGOS_OP_mm__out": "ascend"},
            use_out=True,
        )
        assert result.returncode == 0
