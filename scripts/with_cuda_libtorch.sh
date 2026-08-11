#!/usr/bin/env bash
# 在进程内外挂版本匹配的 libtorch_cuda.so 后，透传执行任意命令。
#
# 让 torch_fl 的 CUDA 后端（boxing 写法）复用 PyTorch 已注册的 CUDA kernel，
# 而无需 pip 安装 CUDA 版 torch。详见 docs/vendors/cuda/external-libtorch-cuda.md。
#
# 硬约束（docs §约束1）：libtorch_cuda.so 必须在 `import torch` 之前载入
# （CUDAHooks 缓存问题），因此这里用 LD_PRELOAD 注入，而不是在 __init__.py 里后加载。
#
# 用法:
#   bash scripts/with_cuda_libtorch.sh pytest tests/integration/ops/test_add_dispatch.py -v
#   bash scripts/with_cuda_libtorch.sh python -c "import torch_fl, torch; ..."

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ASSETS="${REPO_DIR}/.libtorch_cuda_assets"

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/with_cuda_libtorch.sh <command> [args...]" >&2
  exit 2
fi

for _so in libc10_cuda.so libtorch_cuda.so; do
  if [ ! -f "${CUDA_ASSETS}/${_so}" ]; then
    echo "error: missing ${CUDA_ASSETS}/${_so}" >&2
    echo "  (see docs/vendors/cuda/external-libtorch-cuda.md for how these assets are produced)" >&2
    exit 1
  fi
done

# 1) nvidia runtime 库路径 + pip torch 的 lib 目录（libc10_cuda.so 依赖 libc10.so）
#    + CUDA_ASSETS 本身（linalg 等算子会按裸名 dlopen libtorch_cuda_linalg.so，
#      该 .so 就放在 CUDA_ASSETS 里，必须在 LD_LIBRARY_PATH 上才能被找到）。
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="${CUDA_ASSETS}:$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr '\n' ':')${TORCH_LIB}:${LD_LIBRARY_PATH}"

# 2) 硬约束：在 import torch 之前把 CUDA .so 载入进程 -> LD_PRELOAD。
export LD_PRELOAD="${CUDA_ASSETS}/libc10_cuda.so:${CUDA_ASSETS}/libtorch_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}"

exec "$@"
