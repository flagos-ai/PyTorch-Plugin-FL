#!/usr/bin/env bash
# 把 PPU 本地构建的 libtorch C++ .so 打进 torch_fl/lib_ppu/，做成自包含单 wheel。
#
# PPU 是对着 PPU_SDK/CUDA_SDK 编的，所以 ACCELERATOR=cuda、CUDA boxing kernel
# 原样可用。跟真 NVIDIA 机器的区别在 libtorch：它是本地 USE_CUDA=1 源码构建，不是
# 上游 wheel。实测 libtorch_fl.so 的未定义符号里它的 libtorch_cpu.so 提供 2092 个，
# libtorch_cuda.so 提供 0 个、libc10_cuda.so 10 个 —— 所以 core 库必须换掉，
# libtorch_cuda.so 只需在场（CUDA dispatch key 上要有已注册的厂商 kernel）。
#
# 那个本地构建还链了 /usr/local/lib 下的系统 MKL
# （libmkl_core / libmkl_gnu_thread / libmkl_intel_lp64，~171 MB），官方
# torch+cpu wheel 里没有对应文件，所以一并打进 lib_ppu/。
#
# 不打包：PPU SDK runtime 留在目标机 /usr/local/PPU_SDK。
#
# 用法:
#   FLAGOS_PPU_TORCH_LIB=<ppu torch/lib> bash scripts/bundle_ppu_libtorch.sh
#   PPU_SDK=/usr/local/PPU_SDK bash scripts/bundle_ppu_libtorch.sh
#   FLAGOS_PPU_MKL_DIR=/usr/local/lib bash scripts/bundle_ppu_libtorch.sh
#
# 应在 `python setup.py bdist_wheel` 之后、打 wheel 之前跑。幂等。

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bundle_common.sh
source "${REPO_DIR}/scripts/lib/bundle_common.sh"

LIB_PPU="${REPO_DIR}/torch_fl/lib_ppu"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
PPU_SDK="${PPU_SDK:-${PPU_HOME:-/usr/local/PPU_SDK}}"
MKL_DIR="${FLAGOS_PPU_MKL_DIR:-/usr/local/lib}"

SRC="${FLAGOS_PPU_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  # PPU 的 torch 是本地构建，version.py 里不一定有 "ppu" 字样，所以先按标记找，
  # 找不到就退化成"当前解释器的 torch 只要有 libtorch_cuda.so 就算"。
  SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so ppu || true)"
  if [ -z "${SRC}" ]; then
    SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so || true)"
  fi
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: 找不到 PPU torch/lib。设 FLAGOS_PPU_TORCH_LIB=<ppu torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_cuda.so" ]; then
  echo "error: ${SRC} 里没有 libtorch_cuda.so，不是 CUDA 构建的 torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# 与 _ppu_libtorch_link._CORE_SO + _CUDA_SO 对齐的自洽集合。
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so)
CUDA_SO=(libc10_cuda.so libtorch_cuda.so libtorch_cuda_linalg.so libshm.so)
MKL_SO=(libmkl_core.so.1 libmkl_gnu_thread.so.1 libmkl_intel_lp64.so.1)

VENDOR_RPATH="${PPU_SDK}/CUDA_SDK/lib64:${PPU_SDK}/lib:${PPU_SDK}/lib64"
# 同 DCU：bundle 内的库要能从 lib_ppu/ 和 torch/lib/ 软链两条路径被打开。
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_ppu"

echo "源 PPU torch/lib : ${SRC}"
echo "目标 lib_ppu     : ${LIB_PPU}"
echo "PPU SDK 路径     : ${PPU_SDK}"

bundle_copy_so "${SRC}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 1 "${CORE_SO[@]}"
bundle_copy_so "${SRC}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${CUDA_SO[@]}"

# 系统 MKL：libtorch_cpu.so 的直接 DT_NEEDED，官方 wheel 里没有同名文件。
if [ -d "${MKL_DIR}" ]; then
  echo "源 MKL           : ${MKL_DIR}"
  bundle_copy_so "${MKL_DIR}" "${LIB_PPU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${MKL_SO[@]}"
else
  echo "warning: ${MKL_DIR} 不存在，跳过 MKL；目标机上若无 MKL 会缺库" >&2
fi

bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" \
    "\$ORIGIN:\$ORIGIN/../lib_ppu:${VENDOR_RPATH}"

bundle_summary "${LIB_PPU}"
bundle_check_needed "${LIB_PPU}" \
    "${PPU_SDK}/CUDA_SDK/lib64" "${PPU_SDK}/lib" "${PPU_SDK}/lib64" \
    "${MKL_DIR}" "/usr/lib64" "/usr/lib" "/usr/local/lib"
