#!/usr/bin/env python3
"""Test codegen'd mm/bmm (functional + .out variants) vs CPU.

mm  -> aclnnMm,          mm.out  -> aclnnMm  (MmOutFn)
bmm -> aclnnBatchMatMul, bmm.out -> aclnnBatchMatMul (BmmOutFn)
"""

import os
os.environ['FLAGOS_BACKEND_CONFIG'] = 'torch_fl/backends_ascend.conf'

import torch
import torch_fl  # noqa: F401

device = torch.device('privateuseone:0')


# f32 matmul uses hf32 cube accumulation (allow_hf32=true, same as the original
# handwritten kernel), so rel-err lands around 1e-2 rather than 1e-4.
def _check(name, got, ref, atol=2e-2):
    diff = (got.float().cpu() - ref.float()).abs().max().item()
    ok = diff <= atol
    print(f"  {'✓' if ok else '✗'} {name}: max abs diff {diff:g}")
    assert ok, f"{name} mismatch (diff={diff})"


def test_mm():
    print("\n=== mm (aclnnMm) ===")
    a = torch.randn(32, 48, dtype=torch.float32)
    b = torch.randn(48, 24, dtype=torch.float32)
    ref = a @ b
    _check("mm f32", torch.mm(a.to(device), b.to(device)), ref)

    ah = a.half(); bh = b.half()
    _check("mm f16", torch.mm(ah.to(device), bh.to(device)), (ah @ bh).float(), atol=5e-1)


def test_mm_out():
    print("\n=== mm.out (MmOutFn) ===")
    a = torch.randn(16, 20, dtype=torch.float32)
    b = torch.randn(20, 12, dtype=torch.float32)
    ref = a @ b
    out = torch.empty(16, 12, dtype=torch.float32, device=device)
    r = torch.mm(a.to(device), b.to(device), out=out)
    _check("mm.out result", r, ref)
    _check("mm.out aliases out", out, ref)
    assert r.data_ptr() == out.data_ptr(), "mm.out must write into provided out"


def test_bmm():
    print("\n=== bmm (aclnnBatchMatMul) ===")
    a = torch.randn(8, 32, 48, dtype=torch.float32)
    b = torch.randn(8, 48, 24, dtype=torch.float32)
    ref = torch.bmm(a, b)
    _check("bmm f32", torch.bmm(a.to(device), b.to(device)), ref)


def test_bmm_out():
    print("\n=== bmm.out (BmmOutFn) ===")
    a = torch.randn(4, 10, 16, dtype=torch.float32)
    b = torch.randn(4, 16, 7, dtype=torch.float32)
    ref = torch.bmm(a, b)
    out = torch.empty(4, 10, 7, dtype=torch.float32, device=device)
    r = torch.bmm(a.to(device), b.to(device), out=out)
    _check("bmm.out result", r, ref)
    _check("bmm.out aliases out", out, ref)
    assert r.data_ptr() == out.data_ptr(), "bmm.out must write into provided out"


if __name__ == '__main__':
    print("Testing codegen'd mm/bmm (+ .out) on Ascend NPU...")
    try:
        test_mm()
        test_mm_out()
        test_bmm()
        test_bmm_out()
        print("\n" + "=" * 60)
        print("ALL MATMUL CODEGEN TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
