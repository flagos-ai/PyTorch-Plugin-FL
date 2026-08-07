#!/usr/bin/env bash
# 把沐曦 fork 的 libtorch C++ .so 打进 torch_fl/lib_maca/，做成自包含单 wheel。
#
# 官方 torch+cpu wheel 缺沐曦 fork 的 libtorch（带 at::maca::* / wcuda* 符号），
# boxing 产物 libtorch_fl.so 又必须链接这些符号。本脚本从沐曦 torch wheel 拷贝那批
# libtorch .so 到 torch_fl/lib_maca/，并用 patchelf 重写 RPATH：
#   - lib_maca 内 libtorch .so     -> $ORIGIN:/opt/maca/lib:/opt/maca/lib64
#     （运行期从目标机 /opt/maca 找 mcblas/mcdnn 等 maca runtime；不打包 runtime）
#   - torch_fl/lib/libtorch_fl.so  -> $ORIGIN:$ORIGIN/../lib_maca
#     （从包内 lib_maca 找 fork libtorch，去掉构建机写死的绝对路径）
#
# maca runtime（libmcblas 等，~4.9G）不打包：装了沐曦卡的机器必有 /opt/maca 驱动。
#
# 用法:
#   FLAGOS_MACA_TORCH_LIB=<maca torch/lib> bash scripts/bundle_maca_libtorch.sh
#   MACA_PATH=/opt/maca bash scripts/bundle_maca_libtorch.sh   # 覆盖 maca 路径
#
# 应在 `python setup.py bdist_wheel`（ACCELERATOR=metax）之后、打 wheel 之前跑，
# 或跑完再重打 wheel。幂等。

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bundle_common.sh
source "${REPO_DIR}/scripts/lib/bundle_common.sh"

LIB_MACA="${REPO_DIR}/torch_fl/lib_maca"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
MACA_PATH="${MACA_PATH:-${METAX_HOME:-${MACA_HOME:-/opt/maca}}}"

# 沐曦 torch/lib 来源：显式 env，或从 conda 里找 +metax torch。
SRC="${FLAGOS_MACA_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  SRC="$(bundle_find_vendor_torch_lib libtorch_cuda.so metax maca || true)"
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: 找不到沐曦 torch/lib。设 FLAGOS_MACA_TORCH_LIB=<maca torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_cuda.so" ]; then
  echo "error: ${SRC} 里没有 libtorch_cuda.so，不是沐曦 torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# 与 _metax_libtorch_link._CORE_SO + _CUDA_SO 对齐的自洽集合。
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so)
CUDA_SO=(libc10_cuda.so libtorch_cuda.so libtorch_cuda_linalg.so libshm.so)

VENDOR_RPATH="${MACA_PATH}/lib:${MACA_PATH}/lib64"
# 同 DCU：bundle 内的库要能从 lib_maca/ 和 torch/lib/ 软链两条路径被打开。
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_maca"

echo "源沐曦 torch/lib : ${SRC}"
echo "目标 lib_maca    : ${LIB_MACA}"
echo "maca runtime 路径: ${MACA_PATH}/lib"

bundle_copy_so "${SRC}" "${LIB_MACA}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 \
    "${CORE_SO[@]}" "${CUDA_SO[@]}"

# torch_fl.so 去掉构建机写死的沐曦 torch/lib 绝对路径，改为从包内 lib_maca 找 fork libtorch。
bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" \
    "\$ORIGIN:\$ORIGIN/../lib_maca:${VENDOR_RPATH}"

bundle_summary "${LIB_MACA}"
bundle_check_needed "${LIB_MACA}" \
    "${MACA_PATH}/lib" "${MACA_PATH}/lib64" "/usr/lib64" "/usr/lib"
