#!/usr/bin/env python3
"""Test high-level F.scaled_dot_product_attention on Ascend NPU."""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch.nn.functional as F
import torch_fl

device = torch.device('privateuseone:0')

def test_sdpa_highlevel_forward():
    """Test F.scaled_dot_product_attention forward."""
    print("\n=== Test F.scaled_dot_product_attention (forward) ===")

    B, N, S, D = 2, 4, 128, 64

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    out = F.scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, N, S, D)
    assert out.float().cpu().abs().max().item() > 0
    print("✓ Forward successful")

def test_sdpa_highlevel_causal():
    """Test F.scaled_dot_product_attention with causal mask."""
    print("\n=== Test F.scaled_dot_product_attention (causal) ===")

    B, N, S, D = 2, 4, 128, 64

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"Output shape: {out.shape}")
    assert out.shape == (B, N, S, D)
    assert out.float().cpu().abs().max().item() > 0
    print("✓ Causal forward successful")

def test_sdpa_highlevel_backward():
    """Test F.scaled_dot_product_attention with autograd backward."""
    print("\n=== Test F.scaled_dot_product_attention (backward) ===")

    B, N, S, D = 2, 4, 128, 64

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device).requires_grad_(True)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device).requires_grad_(True)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device).requires_grad_(True)

    out = F.scaled_dot_product_attention(q, k, v)
    # provide explicit grad (ones_like -> fill_.Scalar not registered on Ascend)
    grad_out = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    out.backward(grad_out)

    print(f"grad_q norm: {q.grad.float().cpu().norm().item():.6f}")
    print(f"grad_k norm: {k.grad.float().cpu().norm().item():.6f}")
    print(f"grad_v norm: {v.grad.float().cpu().norm().item():.6f}")

    assert q.grad is not None and q.grad.float().cpu().abs().max().item() > 0
    assert k.grad is not None and k.grad.float().cpu().abs().max().item() > 0
    assert v.grad is not None and v.grad.float().cpu().abs().max().item() > 0
    print("✓ Backward successful")

def test_sdpa_correctness():
    """Compare against CPU reference (naive attention)."""
    print("\n=== Test SDPA correctness vs CPU reference ===")

    B, N, S, D = 1, 2, 64, 32

    q_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
    k_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
    v_cpu = torch.randn(B, N, S, D, dtype=torch.float32)

    # CPU reference
    ref = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)

    # NPU (fp16)
    q = q_cpu.half().to(device)
    k = k_cpu.half().to(device)
    v = v_cpu.half().to(device)
    out = F.scaled_dot_product_attention(q, k, v)
    out_cpu = out.float().cpu()

    max_diff = (out_cpu - ref).abs().max().item()
    rel_diff = max_diff / (ref.abs().max().item() + 1e-8)
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Max rel diff: {rel_diff:.6f}")

    # fp16 tolerance
    assert rel_diff < 0.05, f"Relative difference too large: {rel_diff}"
    print("✓ Correctness verified (within fp16 tolerance)")

if __name__ == '__main__':
    print("Testing high-level SDPA on Ascend NPU...")

    try:
        test_sdpa_highlevel_forward()
        test_sdpa_highlevel_causal()
        test_sdpa_highlevel_backward()
        test_sdpa_correctness()

        print("\n" + "="*60)
        print("ALL HIGH-LEVEL SDPA TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
