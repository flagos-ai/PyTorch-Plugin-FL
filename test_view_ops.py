#!/usr/bin/env python3
"""Test that view ops work on Ascend NPU."""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch_fl

device = torch.device('privateuseone:0')

def test_transpose():
    """Test transpose.int"""
    print("\n=== Test transpose.int ===")
    x = torch.randn(2, 3, 4, 5).to(device)
    y = x.transpose(1, 2)  # Swap dims 1 and 2

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4, 3, 5), f"Expected (2,4,3,5), got {y.shape}"
    print("✓ transpose works")

def test_permute():
    """Test permute"""
    print("\n=== Test permute ===")
    x = torch.randn(2, 3, 4, 5).to(device)
    y = x.permute(0, 2, 1, 3)  # Reorder to [B, S, N, D]

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4, 3, 5), f"Expected (2,4,3,5), got {y.shape}"
    print("✓ permute works")

def test_select():
    """Test select.int"""
    print("\n=== Test select.int ===")
    x = torch.randn(2, 3, 4, 5).to(device)
    y = x.select(1, 0)  # Select first element along dim 1

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 4, 5), f"Expected (2,4,5), got {y.shape}"
    print("✓ select works")

def test_slice():
    """Test slice.Tensor"""
    print("\n=== Test slice.Tensor ===")
    x = torch.randn(2, 3, 8, 5).to(device)
    y = x[:, :, 2:6, :]  # Slice dim 2 from index 2 to 6

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 3, 4, 5), f"Expected (2,3,4,5), got {y.shape}"
    print("✓ slice works")

def test_squeeze():
    """Test squeeze and squeeze.dim"""
    print("\n=== Test squeeze ===")
    x = torch.randn(2, 1, 4, 1, 5).to(device)
    y = x.squeeze()  # Remove all size-1 dims

    print(f"Input shape: {x.shape}")
    print(f"Output shape (squeeze all): {y.shape}")
    assert y.shape == (2, 4, 5), f"Expected (2,4,5), got {y.shape}"

    z = x.squeeze(1)  # Remove only dim 1
    print(f"Output shape (squeeze dim 1): {z.shape}")
    assert z.shape == (2, 4, 1, 5), f"Expected (2,4,1,5), got {z.shape}"
    print("✓ squeeze works")

def test_unsqueeze():
    """Test unsqueeze"""
    print("\n=== Test unsqueeze ===")
    x = torch.randn(2, 3, 4).to(device)
    y = x.unsqueeze(1)  # Add dim at position 1

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == (2, 1, 3, 4), f"Expected (2,1,3,4), got {y.shape}"
    print("✓ unsqueeze works")

if __name__ == '__main__':
    print("Testing view ops on Ascend NPU...")

    try:
        test_transpose()
        test_permute()
        test_select()
        test_slice()
        test_squeeze()
        test_unsqueeze()

        print("\n" + "="*60)
        print("ALL VIEW OPS TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
