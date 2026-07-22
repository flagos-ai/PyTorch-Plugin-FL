#!/usr/bin/env python3
"""Test high-level F.scaled_dot_product_attention API on Ascend NPU."""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch.nn.functional as F
import torch_fl

device = torch.device('privateuseone:0')

def test_sdpa_efficient_backend():
    """Test that F.scaled_dot_product_attention routes to efficient_attention backend."""
    print("\n=== Test F.scaled_dot_product_attention (high-level API) ===")

    B, N, S, D = 2, 4, 128, 64

    # Create inputs with requires_grad for autograd test
    q = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)

    # Call high-level API
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=False,
        enable_mem_efficient=True
    ):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

    print(f"Forward output shape: {out.shape}")
    print(f"Output norm: {out.float().cpu().norm().item():.6f}")

    # Test backward
    loss = out.sum()
    loss.backward()

    print(f"grad_q norm: {q.grad.float().cpu().norm().item():.6f}")
    print(f"grad_k norm: {k.grad.float().cpu().norm().item():.6f}")
    print(f"grad_v norm: {v.grad.float().cpu().norm().item():.6f}")

    assert q.grad.float().cpu().abs().max().item() > 0
    assert k.grad.float().cpu().abs().max().item() > 0
    assert v.grad.float().cpu().abs().max().item() > 0

    print("✓ High-level API forward+backward successful")

def test_sdpa_causal():
    """Test causal attention with high-level API."""
    print("\n=== Test F.scaled_dot_product_attention (causal) ===")

    B, N, S, D = 2, 4, 128, 64

    q = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16, requires_grad=True).to(device)

    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=False,
        enable_mem_efficient=True
    ):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    loss = out.sum()
    loss.backward()

    print(f"grad_q norm: {q.grad.float().cpu().norm().item():.6f}")
    print(f"grad_k norm: {k.grad.float().cpu().norm().item():.6f}")
    print(f"grad_v norm: {v.grad.float().cpu().norm().item():.6f}")

    assert q.grad.float().cpu().abs().max().item() > 0

    print("✓ Causal attention successful")

if __name__ == '__main__':
    print("Testing high-level SDPA API on Ascend NPU...")

    try:
        test_sdpa_efficient_backend()
        test_sdpa_causal()

        print("\n" + "="*60)
        print("HIGH-LEVEL API TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
