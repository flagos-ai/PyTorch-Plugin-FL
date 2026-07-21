#!/usr/bin/env python3
"""Test SDPA Ascend kernel (forward only, backward blocked by logsumexp issue)"""

import torch
import torch_fl
import os

os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

def test_sdpa_forward_noncausal():
    """Test non-causal SDPA forward"""
    print("\n=== Test SDPA Forward (non-causal) ===")

    B, N, S, D = 2, 8, 128, 64

    # CPU reference
    q_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
    k_cpu = q_cpu.clone()
    v_cpu = q_cpu.clone()

    # Manually compute attention on CPU
    scale = 1.0 / (D ** 0.5)
    scores_cpu = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale
    attn_cpu = torch.nn.functional.softmax(scores_cpu, dim=-1)
    out_cpu = torch.matmul(attn_cpu, v_cpu)

    # NPU via SDPA
    q_npu = q_cpu.to('privateuseone')
    k_npu = k_cpu.to('privateuseone')
    v_npu = v_cpu.to('privateuseone')

    # Call _scaled_dot_product_efficient_attention directly
    out_npu, logsumexp, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        q_npu, k_npu, v_npu,
        None,  # attn_bias
        True,  # compute_log_sumexp
        0.0,   # dropout_p
        False, # is_causal
        scale=None  # keyword-only
    )

    out_npu_cpu = out_npu.cpu()
    diff = (out_npu_cpu - out_cpu).abs()
    max_err = diff.max().item()

    print(f"  Input shape: [{B}, {N}, {S}, {D}]")
    print(f"  Output shape: {out_npu.shape}")
    print(f"  logsumexp shape: {logsumexp.shape}")
    print(f"  Max error vs CPU: {max_err:.2e}")

    if max_err < 1e-3:
        print("  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL (err {max_err:.2e} > 1e-3)")
        return False


def test_sdpa_forward_causal():
    """Test causal SDPA forward"""
    print("\n=== Test SDPA Forward (causal) ===")

    B, N, S, D = 1, 4, 64, 32

    # CPU reference with causal mask
    q_cpu = torch.randn(B, N, S, D, dtype=torch.float32)
    k_cpu = q_cpu.clone()
    v_cpu = q_cpu.clone()

    scale = 1.0 / (D ** 0.5)
    scores_cpu = torch.matmul(q_cpu, k_cpu.transpose(-2, -1)) * scale

    # Apply causal mask (upper triangular = -inf)
    mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
    scores_cpu = scores_cpu.masked_fill(mask, float('-inf'))

    attn_cpu = torch.nn.functional.softmax(scores_cpu, dim=-1)
    out_cpu = torch.matmul(attn_cpu, v_cpu)

    # NPU via SDPA
    q_npu = q_cpu.to('privateuseone')
    k_npu = k_cpu.to('privateuseone')
    v_npu = v_cpu.to('privateuseone')

    out_npu, logsumexp, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        q_npu, k_npu, v_npu,
        None,  # attn_bias
        True,  # compute_log_sumexp
        0.0,   # dropout_p
        True,  # is_causal ← TRUE
        scale=None  # keyword-only
    )

    out_npu_cpu = out_npu.cpu()
    diff = (out_npu_cpu - out_cpu).abs()
    max_err = diff.max().item()

    print(f"  Input shape: [{B}, {N}, {S}, {D}]")
    print(f"  Output shape: {out_npu.shape}")
    print(f"  logsumexp shape: {logsumexp.shape}")
    print(f"  Max error vs CPU causal: {max_err:.2e}")

    if max_err < 1e-3:
        print("  ✓ PASS")
        return True
    else:
        print(f"  ✗ FAIL (err {max_err:.2e} > 1e-3)")
        return False


def test_sdpa_backward_blocked():
    """Test that backward correctly raises the NotImplemented error"""
    print("\n=== Test SDPA Backward (expect error) ===")

    B, N, S, D = 1, 2, 32, 16

    q_npu = torch.randn(B, N, S, D, requires_grad=True).to('privateuseone')
    k_npu = torch.randn(B, N, S, D, requires_grad=True).to('privateuseone')
    v_npu = torch.randn(B, N, S, D, requires_grad=True).to('privateuseone')

    try:
        out, logsumexp, seed, offset = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q_npu, k_npu, v_npu, None, True, 0.0, False, scale=None)

        # Try backward - should fail because backward kernel is not implemented
        grad_out = torch.ones(out.shape, dtype=out.dtype).to(out.device)
        try:
            # Manually call backward kernel
            torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
                grad_out, q_npu, k_npu, v_npu, None, out, logsumexp, seed, offset,
                0.0, [True, True, True, False], False, scale=None)
            print("  ✗ FAIL: backward succeeded (should have raised NotImplementedError)")
            return False
        except RuntimeError as bwd_err:
            if "backend not registered" in str(bwd_err) or "not implemented" in str(bwd_err).lower():
                print(f"  ✓ PASS: backward correctly blocked")
                print(f"     Error: {str(bwd_err)[:80]}...")
                return True
            else:
                print(f"  ✗ FAIL: unexpected backward error: {bwd_err}")
                return False

    except RuntimeError as e:
        print(f"  ✗ FAIL: forward failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SDPA Ascend Kernel Test")
    print("=" * 60)

    results = []

    # Forward tests
    results.append(("SDPA forward non-causal", test_sdpa_forward_noncausal()))
    results.append(("SDPA forward causal", test_sdpa_forward_causal()))
    results.append(("SDPA backward blocked", test_sdpa_backward_blocked()))

    print("\n" + "=" * 60)
    print("Summary:")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    print(f"\nTotal: {passed}/{total} passed")
    print("=" * 60)

    exit(0 if passed == total else 1)
