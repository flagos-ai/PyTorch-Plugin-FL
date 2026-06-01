"""Test which ops work via flagos_python (FlagGems Triton) on Ascend."""
import os
os.environ.setdefault("FLAGOS_BACKEND_CONFIG", "torch_fl/backends_ascend_flagos_py.conf")

import torch
import torch_fl

device = torch.device("flagos:0")
x = torch.randn(2, 3, device=device)
y = torch.randn(2, 3, device=device)
xi = torch.tensor([1, 2, 3], device=device)
yi = torch.tensor([3, 2, 1], device=device)

ops = {
    "abs": lambda: torch.abs(x),
    "acos": lambda: torch.acos(x.clamp(-1, 1)),
    "add.Tensor": lambda: x + y,
    "all": lambda: torch.all(x > 0),
    "bitwise_and.Tensor": lambda: torch.bitwise_and(xi, yi),
    "bmm": lambda: torch.bmm(x.unsqueeze(0), y.unsqueeze(0).transpose(1, 2)),
    "cat": lambda: torch.cat([x, y], dim=0),
    "constant_pad_nd": lambda: torch.nn.functional.pad(x, (1, 1)),
    "cos": lambda: torch.cos(x),
    "le.Tensor": lambda: x <= y,
    "mean.dim": lambda: x.mean(dim=1),
    "mm": lambda: torch.mm(x.T, y),
    "mul.Tensor": lambda: x * y,
    "mul.Scalar": lambda: x * 2.0,
    "neg": lambda: torch.neg(x),
    "pow.Tensor_Scalar": lambda: torch.pow(x, 2.0),
    "rsqrt": lambda: torch.rsqrt(torch.abs(x) + 1e-6),
    "silu": lambda: torch.nn.functional.silu(x),
    "sin": lambda: torch.sin(x),
    "_softmax": lambda: torch.softmax(x, dim=1),
    "sum.dim_IntList": lambda: x.sum(dim=1),
    "where.self": lambda: torch.where(x > 0, x, y),
    "index.Tensor": lambda: x[torch.tensor([0, 1], device=device)],
}

passed = []
failed = []

for name, fn in ops.items():
    try:
        result = fn()
        torch_fl.flagos.synchronize()
        passed.append(name)
        print(f"PASS: {name}")
    except Exception as e:
        err = str(e).split("\n")[0][:120]
        failed.append((name, err))
        print(f"FAIL: {name} -> {err}")

print(f"\n{'='*60}")
print(f"PASSED: {len(passed)}/{len(ops)}")
print(f"FAILED: {len(failed)}/{len(ops)}")
if failed:
    print("\nFailed ops:")
    for name, err in failed:
        print(f"  {name}: {err}")
