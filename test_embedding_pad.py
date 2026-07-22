#!/usr/bin/env python3
"""Test codegen'd embedding / embedding_dense_backward / constant_pad_nd vs CPU."""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch_fl  # noqa: F401

device = torch.device('privateuseone:0')


def _check(name, got, ref, atol=1e-4):
    diff = (got.float().cpu() - ref.float()).abs().max().item()
    ok = diff <= atol
    print(f"  {'✓' if ok else '✗'} {name}: max abs diff {diff:g}")
    assert ok, f"{name} mismatch (diff={diff})"


def test_embedding():
    print("\n=== embedding (aclnnEmbedding) ===")
    num_emb, emb_dim = 10, 4
    weight = torch.randn(num_emb, emb_dim, dtype=torch.float32)
    idx = torch.tensor([[1, 3, 5], [0, 9, 2]], dtype=torch.int64)

    ref = torch.embedding(weight, idx)
    got = torch.embedding(weight.to(device), idx.to(device))
    _check("embedding 2d idx", got, ref)

    idx1 = torch.tensor([7, 4, 4, 0], dtype=torch.int64)
    _check("embedding 1d idx",
           torch.embedding(weight.to(device), idx1.to(device)),
           torch.embedding(weight, idx1))


def test_embedding_backward():
    print("\n=== embedding_dense_backward (aclnnEmbeddingDenseBackward) ===")
    num_emb, emb_dim = 8, 5
    idx = torch.tensor([1, 3, 3, 0, 7], dtype=torch.int64)
    grad = torch.randn(5, emb_dim, dtype=torch.float32)

    # CPU reference via autograd
    w_cpu = torch.randn(num_emb, emb_dim, requires_grad=True)
    out_cpu = torch.embedding(w_cpu, idx)
    out_cpu.backward(grad)
    ref = w_cpu.grad

    got = torch.ops.aten.embedding_dense_backward(
        grad.to(device), idx.to(device), num_emb, -1, False)
    _check("embedding_dense_backward", got, ref)


def test_constant_pad_nd():
    print("\n=== constant_pad_nd (aclnnConstantPadNd) ===")
    x = torch.randn(2, 3, 4, dtype=torch.float32)

    # pad last dim (1,2), then also second-to-last (1,1)
    ref = torch.constant_pad_nd(x, [1, 2], 0.0)
    got = torch.constant_pad_nd(x.to(device), [1, 2], 0.0)
    _check("pad last dim, value 0", got, ref)

    ref2 = torch.constant_pad_nd(x, [1, 2, 1, 1], 3.5)
    got2 = torch.constant_pad_nd(x.to(device), [1, 2, 1, 1], 3.5)
    _check("pad two dims, value 3.5", got2, ref2)


if __name__ == '__main__':
    print("Testing embedding / embedding_backward / constant_pad_nd on Ascend NPU...")
    try:
        test_embedding()
        test_embedding_backward()
        test_constant_pad_nd()
        print("\n" + "=" * 60)
        print("ALL EMBEDDING / PAD TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
