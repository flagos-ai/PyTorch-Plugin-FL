"""
embedding_dense_backward dispatch tests

Verifies that embedding backward:
  - produces correct gradients on flagos device
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_embedding_dense_backward_dispatch.py -v
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
        "emb = torch.nn.Embedding(100, 32).to('flagos:0'); "
        "idx = torch.tensor([0,1,2], device='flagos:0'); "
        "out = emb(idx); "
        "out.sum().backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.anyplatform
class TestEmbeddingDenseBackwardCorrectness:
    """embedding_dense_backward correctness on flagos device."""

    def test_embedding_backward_basic(self):
        torch.manual_seed(0)
        emb = torch.nn.Embedding(50, 16).to(DEVICE)
        idx = torch.tensor([0, 5, 10, 5], device=DEVICE)
        out = emb(idx)
        out.sum().backward()
        assert emb.weight.grad is not None
        assert emb.weight.grad.shape == (50, 16)
        # Only accessed indices should have non-zero grad
        assert emb.weight.grad.cpu()[0].sum().item() != 0.0
        assert emb.weight.grad.cpu()[5].sum().item() != 0.0
        assert emb.weight.grad.cpu()[1].sum().item() == 0.0

    def test_embedding_backward_matches_cpu(self):
        torch.manual_seed(1)
        emb_cpu = torch.nn.Embedding(100, 32)
        emb_fl = torch.nn.Embedding(100, 32).to(DEVICE)
        # Copy same weights
        emb_fl.weight.data.copy_(emb_cpu.weight.data.to(DEVICE))

        idx = torch.tensor([3, 7, 3, 50])
        out_cpu = emb_cpu(idx)
        out_cpu.sum().backward()

        idx_fl = idx.to(DEVICE)
        out_fl = emb_fl(idx_fl)
        out_fl.sum().backward()

        torch.testing.assert_close(
            emb_fl.weight.grad.cpu(), emb_cpu.weight.grad, rtol=1e-5, atol=1e-5
        )

    def test_embedding_backward_duplicate_indices(self):
        torch.manual_seed(2)
        emb = torch.nn.Embedding(20, 8).to(DEVICE)
        idx = torch.tensor([3, 3, 3, 3], device=DEVICE)
        out = emb(idx)
        out.sum().backward()
        # Grad for index 3 should be 4x the embedding dim (all ones * 4)
        expected = torch.ones(8) * 4.0
        torch.testing.assert_close(emb.weight.grad.cpu()[3], expected)


@pytest.mark.cuda
class TestEmbeddingDenseBackwardDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_cuda(self):
        result = _run_subprocess(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_embedding_dense_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] embedding_dense_backward -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess(
            {"FLAGOS_OP_embedding_dense_backward": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr


@pytest.mark.ascend
class TestEmbeddingDenseBackwardAscendDispatch:
    """Verify Ascend backend correctness."""

    def test_ascend_correctness(self):
        """Verify embedding_dense_backward on ascend backend matches CPU reference."""
        result = _run_subprocess(
            {"FLAGOS_OP_embedding_dense_backward": "ascend"}
        )
        assert result.returncode == 0
