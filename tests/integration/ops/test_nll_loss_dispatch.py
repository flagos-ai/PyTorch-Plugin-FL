"""
nll_loss_forward and nll_loss_backward dispatch tests

Verifies that torch.nn.functional.nll_loss:
  - produces correct results on flagos device
  - backward produces correct gradients
  - C++ wrapper routes to cuda backend

Usage:
    pytest tests/integration/ops/test_nll_loss_dispatch.py -v
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_fl  # noqa: F401


DEVICE = "flagos:0"


def _run_subprocess_forward(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "inp = torch.randn(4,10,device='flagos:0').log_softmax(dim=1); "
        "target = torch.tensor([0,3,5,9], device='flagos:0'); "
        "F.nll_loss(inp, target)"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def _run_subprocess_backward(
    extra_env: dict, check: bool = True
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(extra_env)
    code = (
        "import torch_fl, torch, torch.nn.functional as F; "
        "inp = torch.randn(4,10,device='flagos:0',requires_grad=True).log_softmax(dim=1); "
        "target = torch.tensor([0,3,5,9], device='flagos:0'); "
        "loss = F.nll_loss(inp, target); "
        "loss.backward()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


class TestNllLossForwardCorrectness:
    """nll_loss_forward correctness on flagos device."""

    def test_nll_loss_basic(self):
        torch.manual_seed(0)
        inp = torch.randn(8, 10, device=DEVICE).log_softmax(dim=1)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        loss = F.nll_loss(inp, target)
        assert loss.shape == ()
        assert loss.device.type == "flagos"

    def test_nll_loss_matches_cpu(self):
        torch.manual_seed(1)
        inp_cpu = torch.randn(16, 5).log_softmax(dim=1)
        target_cpu = torch.randint(0, 5, (16,))
        ref = F.nll_loss(inp_cpu, target_cpu)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_nll_loss_reduction(self, reduction):
        torch.manual_seed(2)
        inp_cpu = torch.randn(8, 10).log_softmax(dim=1)
        target_cpu = torch.randint(0, 10, (8,))
        ref = F.nll_loss(inp_cpu, target_cpu, reduction=reduction)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl, reduction=reduction)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)

    def test_nll_loss_ignore_index(self):
        torch.manual_seed(3)
        inp_cpu = torch.randn(8, 10).log_softmax(dim=1)
        target_cpu = torch.randint(0, 10, (8,))
        target_cpu[0] = -100
        ref = F.nll_loss(inp_cpu, target_cpu, ignore_index=-100)

        inp_fl = inp_cpu.to(DEVICE)
        target_fl = target_cpu.to(DEVICE)
        out = F.nll_loss(inp_fl, target_fl, ignore_index=-100)
        torch.testing.assert_close(out.cpu(), ref, rtol=1e-4, atol=1e-4)


class TestNllLossBackwardCorrectness:
    """nll_loss_backward correctness on flagos device."""

    def test_nll_loss_backward_basic(self):
        torch.manual_seed(0)
        inp = torch.randn(8, 10, device=DEVICE, requires_grad=True)
        log_inp = inp.log_softmax(dim=1)
        target = torch.randint(0, 10, (8,), device=DEVICE)
        loss = F.nll_loss(log_inp, target)
        loss.backward()
        assert inp.grad is not None
        assert inp.grad.shape == (8, 10)

    def test_nll_loss_backward_matches_cpu(self):
        torch.manual_seed(1)
        inp_cpu = torch.randn(16, 5, requires_grad=True)
        log_inp_cpu = inp_cpu.log_softmax(dim=1)
        target_cpu = torch.randint(0, 5, (16,))
        loss_cpu = F.nll_loss(log_inp_cpu, target_cpu)
        loss_cpu.backward()

        torch.manual_seed(1)
        inp_fl = torch.randn(16, 5, device=DEVICE, requires_grad=True)
        log_inp_fl = inp_fl.log_softmax(dim=1)
        target_fl = target_cpu.to(DEVICE)
        loss_fl = F.nll_loss(log_inp_fl, target_fl)
        loss_fl.backward()

        torch.testing.assert_close(
            inp_fl.grad.cpu(), inp_cpu.grad, rtol=1e-4, atol=1e-4
        )


class TestNllLossDispatch:
    """Verify dispatch routing."""

    def test_dispatch_log_forward_cuda(self):
        result = _run_subprocess_forward(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_nll_loss_forward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] nll_loss_forward -> cuda" in result.stderr

    def test_dispatch_log_backward_cuda(self):
        result = _run_subprocess_backward(
            {"FLAGOS_LOG_DISPATCH": "1", "FLAGOS_OP_nll_loss_backward": "cuda"}
        )
        assert result.returncode == 0
        assert "[flagos dispatch] nll_loss_backward -> cuda" in result.stderr

    def test_flaggems_backend_raises_error(self):
        result = _run_subprocess_forward(
            {"FLAGOS_OP_nll_loss_forward": "flaggems"},
            check=False,
        )
        assert result.returncode != 0
        assert "backend not registered" in result.stderr
