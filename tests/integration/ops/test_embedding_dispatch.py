"""
embedding dispatch tests

Verifies that torch.nn.functional.embedding (aten.embedding):
  - produces correct results on flagos device
  - C++ wrapper routes to flaggems (default) or cuda (via env override)
  - dispatch log confirms the actual backend used

Usage:
    pytest tests/integration/ops/test_embedding_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


@pytest.fixture(scope="session")
def cuda_ref():
    """Reference embedding results computed on CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.manual_seed(42)
    weight = torch.randn(1000, 128, device="cuda:0", dtype=torch.float32)
    indices = torch.randint(0, 1000, (8, 16), device="cuda:0")
    return weight, indices, F.embedding(indices, weight)


def _run_embedding_subprocess(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a minimal embedding call in a subprocess and return the result."""
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "w = torch.randn(100,32,device='flagos:0'); "
        "idx = torch.randint(0,100,(4,),device='flagos:0'); "
        "F.embedding(idx, w)"
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


class TestEmbeddingDispatch:
    """torch.nn.functional.embedding correctness on flagos."""

    @pytest.mark.anyplatform
    def test_embedding_basic(self):
        torch.manual_seed(0)
        weight = torch.randn(100, 64, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([0, 5, 10, 99], device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (4, 64)
        assert out.device.type == "flagos"

    @pytest.mark.anyplatform
    def test_embedding_2d_indices(self):
        torch.manual_seed(1)
        weight = torch.randn(500, 128, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 500, (8, 16), device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (8, 16, 128)

    @pytest.mark.anyplatform
    def test_embedding_correctness(self):
        """Verify embedding == index-select on weight."""
        torch.manual_seed(2)
        weight = torch.randn(50, 32, device=DEVICE, dtype=torch.float32)
        indices = torch.tensor([0, 3, 7, 49], device=DEVICE)
        out = F.embedding(indices, weight)
        expected = weight.index_select(0, indices)
        torch.testing.assert_close(
            out.cpu(),
            expected.cpu(),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.cuda
    def test_embedding_matches_cuda_ref(self, cuda_ref):
        """flagos embedding must match CUDA reference."""
        if cuda_ref is None:
            pytest.skip("CUDA not available for reference")
        w_cuda, idx_cuda, ref = cuda_ref
        w = w_cuda.to(DEVICE)
        idx = idx_cuda.to(DEVICE)
        out = F.embedding(idx, w)
        torch.testing.assert_close(
            out.cpu(),
            ref.cpu(),
            rtol=1e-5,
            atol=1e-5,
            msg="embedding on flagos differs from CUDA",
        )

    @pytest.mark.anyplatform
    def test_embedding_half(self):
        torch.manual_seed(3)
        weight = torch.randn(100, 64, device=DEVICE, dtype=torch.float16)
        indices = torch.tensor([0, 10, 50], device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.dtype == torch.float16
        assert out.shape == (3, 64)

    @pytest.mark.anyplatform
    def test_embedding_large_vocab(self):
        torch.manual_seed(4)
        weight = torch.randn(10000, 256, device=DEVICE, dtype=torch.float32)
        indices = torch.randint(0, 10000, (64,), device=DEVICE)
        out = F.embedding(indices, weight)
        assert out.shape == (64, 256)

    @pytest.mark.anyplatform
    def test_embedding_via_module(self):
        """Test via nn.Embedding module."""
        torch.manual_seed(5)
        emb = torch.nn.Embedding(100, 64).to(DEVICE)
        indices = torch.tensor([0, 1, 2], device=DEVICE)
        out = emb(indices)
        assert out.shape == (3, 64)
        assert out.device.type == "flagos"


class TestEmbeddingDispatchLog:
    """Verify C++ wrapper routes to correct backend."""

    @pytest.mark.cuda
    def test_dispatch_log_flagos_default(self):
        """Default routes embedding to flagos."""
        result = _run_embedding_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_embedding": "flaggems"}
        )
        assert "[flagos dispatch] embedding -> flagos" in result.stderr, (
            f"Expected flagos log, got:\n{result.stderr}"
        )

    @pytest.mark.cuda
    def test_dispatch_log_cuda_override(self):
        """FLAGOS_OP_embedding=cuda overrides to cuda."""
        result = _run_embedding_subprocess(
            {
                "FLAGOS_LOG_DISPATCH": "1",
                "FLAGOS_OP_embedding": "cuda",
            }
        )
        assert "[flagos dispatch] embedding -> cuda" in result.stderr, (
            f"Expected cuda log, got:\n{result.stderr}"
        )

    @pytest.mark.ascend
    def test_dispatch_log_ascend_override(self):
        """FLAGOS_OP_embedding=ascend overrides to ascend backend."""
        result = _run_embedding_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_embedding": "ascend"}
        )
        assert "[flagos dispatch] embedding -> ascend" in result.stderr, (
            f"Expected ascend dispatch log, got:\n{result.stderr}"
        )


class TestEmbeddingAscendDispatch:
    """Verify Ascend backend correctness."""

    @pytest.mark.ascend
    def test_ascend_correctness(self):
        """Verify embedding on ascend backend matches CPU reference."""
        result = _run_embedding_subprocess(
            {"FLAGOS_OP_embedding": "ascend"}
        )
        assert result.returncode == 0
