#!/usr/bin/env bash
# 验证 "CPU torch + 外挂 libtorch_cuda.so 复用 CUDA 算子" 方案是否成立。
# 详见 docs/cpu_torch_external_libtorch_cuda.md
#
# 用法（在 2080ti 上）:
#   bash docs/verify_external_cuda.sh
#
# 前置:
#   - conda 环境 libtorch_test 已装 CPU torch 2.13.0 + nvidia-* runtime 包
#   - libtorch_cuda.so 等已固化在 .libtorch_cuda_assets/

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate libtorch_test

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ASSETS="${REPO_DIR}/.libtorch_cuda_assets"

# 1) nvidia runtime 库路径 + pip torch 的 lib 目录（libc10_cuda.so 依赖 libc10.so）
SP=$(python -c 'import site;print(site.getsitepackages()[0])')
TORCH_LIB=$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
export LD_LIBRARY_PATH=$(ls -d "$SP"/nvidia/*/lib | tr '\n' ':')$TORCH_LIB:$LD_LIBRARY_PATH

# 2) 硬约束: libtorch_cuda.so 必须在 import torch 之前载入 -> 用 LD_PRELOAD
#    注意: libc10_cuda.so 依赖 libc10.so（在 TORCH_LIB 中），故上面已把 TORCH_LIB 加入 LD_LIBRARY_PATH
export LD_PRELOAD="${CUDA_ASSETS}/libc10_cuda.so:${CUDA_ASSETS}/libtorch_cuda.so"

python - <<'PY'
import torch
print('torch:', torch.__version__, '(应为 +cpu)')

def has(op, key):
    return torch._C._dispatch_has_kernel_for_dispatch_key(op, key)

# 关卡 2: CUDA kernel 是否注册进 dispatcher
for op in ['aten::mm', 'aten::add.Tensor', 'aten::_softmax', 'aten::bmm']:
    print(f'  {op:18s} CPU={has(op,"CPU")} CUDA={has(op,"CUDA")}')

# torch_fl 走 C++ boxing, 不调 torch.cuda; 纯 python 复现需短路该 gate
torch.cuda._lazy_init = lambda: None
if hasattr(torch.cuda, '_initialized'):
    torch.cuda._initialized = True

def to_cuda(t):
    d = torch.empty(t.shape, dtype=t.dtype, device='cuda')
    d.copy_(t)
    return d

# 关卡 4: 真实计算
a_c, b_c = torch.randn(64, 64), torch.randn(64, 64)
a, b = to_cuda(a_c), to_cuda(b_c)
assert str(a.device) == 'cuda:0', a.device
mm_err = (torch.mm(a, b).cpu() - a_c @ b_c).abs().max().item()
add_err = ((a + b).cpu() - (a_c + b_c)).abs().max().item()
x_c = torch.randn(128, 256); s = torch.softmax(to_cuda(x_c), dim=-1).cpu()
print(f'mm max_err={mm_err:.2e}  add max_err={add_err}  softmax rowsum={s.sum(-1).mean().item():.4f}')
assert mm_err < 1e-4 and add_err == 0.0
print('=== PASS: CPU torch + external libtorch_cuda.so 复用 CUDA 算子成立 ===')
PY
