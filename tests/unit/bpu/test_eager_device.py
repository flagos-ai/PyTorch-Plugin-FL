"""Eager-mode behaviour of the flagos device on an BPU build.

The BPU has no per-op kernels, so eager correctness here is entirely a question
of the *runtime* layer: does UCP-backed memory allocate, does data survive the
round trip through the host mapping, and does every compute op reach the CPU
fallback. Convolution gets its own tests because it does not go through the
generic fallback -- see the BPUWrapperConvolution* wrappers in
csrc/aten/register.cc for why.
"""

from __future__ import annotations

import copy

import pytest
import torch

torch_fl = pytest.importorskip("torch_fl")

pytestmark = pytest.mark.skipif(
    torch_fl._build_accelerator() != "bpu",
    reason="requires a build with ACCELERATOR=bpu",
)

DEV = "flagos"


def test_device_is_available():
    assert torch.accelerator.device_count() >= 1


def test_roundtrip_preserves_data():
    x = torch.randn(64, 64)
    assert torch.equal(x.to(DEV).cpu(), x)


def test_elementwise_matches_cpu():
    a, b = torch.randn(32, 32), torch.randn(32, 32)
    got = (a.to(DEV) + b.to(DEV)) * 2 - 1
    torch.testing.assert_close(got.cpu(), (a + b) * 2 - 1)


def test_matmul_matches_cpu():
    a, b = torch.randn(48, 32), torch.randn(32, 16)
    torch.testing.assert_close(
        (a.to(DEV) @ b.to(DEV)).cpu(), a @ b, atol=1e-5, rtol=1e-5
    )


def test_reduction_and_scalar_readback():
    x = torch.randn(100)
    # .item() goes through _local_scalar_dense, a distinct path from the fallback.
    assert x.to(DEV).sum().item() == pytest.approx(x.sum().item(), abs=1e-4)


def test_conv2d_forward_matches_cpu():
    """aten::convolution on PrivateUse1 routes to convolution_overrideable,
    whose only other kernel raises. Without an explicit wrapper this is a
    NotImplementedError rather than a CPU fallback."""
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).eval()
    x = torch.randn(2, 3, 16, 16)
    expected = conv(x)
    got = copy.deepcopy(conv).to(DEV)(x.to(DEV))
    assert got.device.type == DEV
    torch.testing.assert_close(got.cpu(), expected, atol=1e-5, rtol=1e-5)


def test_conv2d_backward_matches_cpu():
    conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    x_cpu = torch.randn(2, 3, 16, 16, requires_grad=True)
    conv(x_cpu).sum().backward()

    dev_conv = copy.deepcopy(conv)
    dev_conv.zero_grad()
    dev_conv = dev_conv.to(DEV)
    x_dev = x_cpu.detach().clone().to(DEV).requires_grad_(True)
    dev_conv(x_dev).sum().backward()

    torch.testing.assert_close(x_dev.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        dev_conv.weight.grad.cpu(), conv.weight.grad, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        dev_conv.bias.grad.cpu(), conv.bias.grad, atol=1e-4, rtol=1e-4
    )


def test_module_forward_matches_cpu():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 4),
    ).eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        expected = model(x)
        got = copy.deepcopy(model).to(DEV)(x.to(DEV))
    torch.testing.assert_close(got.cpu(), expected, atol=1e-4, rtol=1e-4)


def test_compile_backend_is_registered():
    from torch._dynamo import list_backends

    assert "bpu" in list_backends(exclude_tags=())


def test_compiled_model_matches_eager():
    """Correct with or without hbdk4: partitions that cannot be compiled stay
    on the CPU, so this passes either way -- only the tolerance differs."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
    ).eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(x)
        got = torch.compile(model, backend="bpu")(x)
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
