#!/usr/bin/env python3
"""Test in-place zero_/fill_ (aclnn Inplace ops) and the factory ops that use them.

Verifies the codegen'd zero_ / fill_.Scalar / fill_.Tensor kernels are device-side
(aclnnInplaceZero / aclnnInplaceFillScalar / aclnnInplaceFillTensor) and correct
vs CPU reference.
"""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch_fl  # noqa: F401  (initialize backend)

device = torch.device('privateuseone:0')


def _check(name, got, ref, atol=0):
    got_cpu = got.float().cpu()
    ref_cpu = ref.float()
    diff = (got_cpu - ref_cpu).abs().max().item()
    ok = diff <= atol
    print(f"  {'✓' if ok else '✗'} {name}: max abs diff {diff:g}")
    assert ok, f"{name} mismatch (diff={diff})"


def test_zero_():
    print("\n=== zero_ (aclnnInplaceZero) ===")
    x = torch.randn(3, 4, 5, dtype=torch.float32).to(device)
    x.zero_()
    _check("zero_ f32", x, torch.zeros(3, 4, 5))

    xh = torch.randn(2, 8, dtype=torch.float16).to(device)
    xh.zero_()
    _check("zero_ f16", xh, torch.zeros(2, 8))


def test_fill_scalar():
    print("\n=== fill_.Scalar (aclnnInplaceFillScalar) ===")
    x = torch.randn(3, 4, dtype=torch.float32).to(device)
    x.fill_(2.5)
    _check("fill_ 2.5", x, torch.full((3, 4), 2.5))

    xi = torch.randint(0, 9, (4, 4), dtype=torch.int32).to(device)
    xi.fill_(7)
    _check("fill_ int 7", xi, torch.full((4, 4), 7, dtype=torch.int32))


def test_fill_tensor():
    print("\n=== fill_.Tensor (aclnnInplaceFillTensor) ===")
    x = torch.randn(2, 3, dtype=torch.float32).to(device)
    # 0-dim tensor value on device
    v = torch.tensor(3.14, dtype=torch.float32).to(device)
    x.fill_(v)
    _check("fill_ tensor(3.14)", x, torch.full((2, 3), 3.14), atol=1e-5)


def test_factory_ops():
    print("\n=== factory ops (use zero_/fill_ internally) ===")
    z = torch.zeros(2, 3, 4, device=device)
    _check("zeros", z, torch.zeros(2, 3, 4))

    ref = torch.randn(3, 5).to(device)
    o = torch.ones_like(ref)
    _check("ones_like", o, torch.ones(3, 5))

    n = ref.new_ones(2, 2)
    _check("new_ones", n, torch.ones(2, 2))

    s = torch.scalar_tensor(4.0, device=device)
    _check("scalar_tensor", s, torch.tensor(4.0))


if __name__ == '__main__':
    print("Testing in-place zero_/fill_ + factory ops on Ascend NPU...")
    try:
        test_zero_()
        test_fill_scalar()
        test_fill_tensor()
        test_factory_ops()
        print("\n" + "=" * 60)
        print("ALL IN-PLACE / FACTORY TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
