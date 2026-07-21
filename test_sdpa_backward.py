#!/usr/bin/env python3
"""Test SDPA backward directly via _scaled_dot_product_efficient_attention ops."""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch_fl  # Initialize backend

device = torch.device('privateuseone:0')

def test_forward_backward_direct():
    """Test forward + backward by calling the aten ops directly."""
    print("\n=== Test SDPA Forward+Backward (direct op calls) ===")

    B, N, S, D = 2, 4, 128, 64

    # Create on CPU then move to NPU
    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    # Forward: _scaled_dot_product_efficient_attention
    # Signature: (query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, scale)
    out, logsumexp, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, False, scale=None
    )

    print(f"Forward output shape: {out.shape}")
    print(f"logsumexp shape: {logsumexp.shape}")

    # Backward: _scaled_dot_product_efficient_attention_backward
    # Signature: (grad_out, query, key, value, attn_bias, out, logsumexp,
    #             philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale)
    grad_out = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    grad_q, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out, q, k, v, None, out, logsumexp,
        philox_seed, philox_offset, 0.0, [True, True, True, False], False, scale=None
    )

    print(f"grad_q shape: {grad_q.shape}, norm: {grad_q.float().cpu().norm().item():.6f}")
    print(f"grad_k shape: {grad_k.shape}, norm: {grad_k.float().cpu().norm().item():.6f}")
    print(f"grad_v shape: {grad_v.shape}, norm: {grad_v.float().cpu().norm().item():.6f}")

    assert grad_q.float().cpu().abs().max().item() > 0, "grad_q is all zeros"
    assert grad_k.float().cpu().abs().max().item() > 0, "grad_k is all zeros"
    assert grad_v.float().cpu().abs().max().item() > 0, "grad_v is all zeros"

    print("✓ Forward+Backward successful (no dropout)")

def test_backward_causal_direct():
    """Test backward with causal mask."""
    print("\n=== Test SDPA Backward (causal, direct) ===")

    B, N, S, D = 2, 4, 128, 64

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    out, logsumexp, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, True, scale=None  # is_causal=True
    )

    grad_out = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    grad_q, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out, q, k, v, None, out, logsumexp,
        philox_seed, philox_offset, 0.0, [True, True, True, False], True, scale=None
    )

    print(f"grad_q norm: {grad_q.float().cpu().norm().item():.6f}")
    print(f"grad_k norm: {grad_k.float().cpu().norm().item():.6f}")
    print(f"grad_v norm: {grad_v.float().cpu().norm().item():.6f}")

    assert grad_q.float().cpu().abs().max().item() > 0
    assert grad_k.float().cpu().abs().max().item() > 0
    assert grad_v.float().cpu().abs().max().item() > 0

    print("✓ Backward successful (causal)")

def test_dropout_forward_direct():
    """Test forward with dropout."""
    print("\n=== Test SDPA Forward with Dropout (direct) ===")

    B, N, S, D = 2, 4, 128, 64
    dropout_p = 0.1

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    out1, _, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, dropout_p, False, scale=None
    )
    out2, _, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, dropout_p, False, scale=None
    )

    diff = (out1.float().cpu() - out2.float().cpu()).abs().max().item()
    print(f"Output1 norm: {out1.float().cpu().norm().item():.6f}")
    print(f"Difference between two runs: {diff:.6f}")

    if diff > 1e-5:
        print("✓ Dropout is active (outputs differ between runs)")
    else:
        print("⚠ Warning: outputs identical, dropout may not be applied")

    print("✓ Forward with dropout successful")

def test_backward_dropout_direct():
    """Test backward with dropout."""
    print("\n=== Test SDPA Backward with Dropout (direct) ===")

    B, N, S, D = 2, 4, 128, 64
    dropout_p = 0.1

    q = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    k = torch.randn(B, N, S, D, dtype=torch.float16).to(device)
    v = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    out, logsumexp, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, dropout_p, False, scale=None
    )

    grad_out = torch.randn(B, N, S, D, dtype=torch.float16).to(device)

    grad_q, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
        grad_out, q, k, v, None, out, logsumexp,
        philox_seed, philox_offset, dropout_p, [True, True, True, False], False, scale=None
    )

    print(f"grad_q norm: {grad_q.float().cpu().norm().item():.6f}")
    print(f"grad_k norm: {grad_k.float().cpu().norm().item():.6f}")
    print(f"grad_v norm: {grad_v.float().cpu().norm().item():.6f}")

    assert grad_q.float().cpu().abs().max().item() > 0
    assert grad_k.float().cpu().abs().max().item() > 0
    assert grad_v.float().cpu().abs().max().item() > 0

    print("✓ Backward with dropout successful")

if __name__ == '__main__':
    print("Testing SDPA backward and dropout on Ascend NPU (direct op calls)...")

    try:
        test_forward_backward_direct()
        test_backward_causal_direct()
        # Skip dropout tests (not yet supported by Ascend backend)
        # test_dropout_forward_direct()
        # test_backward_dropout_direct()

        print("\n" + "="*60)
        print("ALL CORE TESTS PASSED ✓")
        print("(dropout not yet supported)")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
