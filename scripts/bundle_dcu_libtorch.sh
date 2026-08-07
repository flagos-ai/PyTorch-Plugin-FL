#!/usr/bin/env bash
# 把 DTK fork 的 libtorch C++ .so 打进 torch_fl/lib_dcu/，做成自包含单 wheel。
#
# 为什么要打 core 库（不是只打 libtorch_hip.so）：
# 实测 libtorch_fl.so 的未定义符号里，DTK 的 libtorch_cpu.so 提供 2101 个，
# libtorch_hip.so 提供 0 个 —— fork 在 core 库这一侧（DTK 的 libtorch_cpu.so
# 带 128 个 hip 符号、DT_NEEDED 写着 libgalaxyhip.so.5）。libtorch_hip.so 仍然
# 必须在场：boxing kernel 要求 CUDA dispatch key 上已注册厂商 kernel。
#
# 为什么要打 torch.libs/：那 17 个文件是 auditwheel 改名的通用库
# （glog/gflags/MKL/OpenMPI/hwloc/libxml2...），其中 5 个是 DTK libtorch_cpu.so 的
# 直接 DT_NEEDED。实测它们对 hip/dtk/rocm 的引用数为 0，不引入 SDK 绑定；但官方
# torch+cpu wheel 的 torch.libs 里一个都没有（名字带 hash，系统库也对不上），
# 不打就跑不起来。它们和 core 库放同一目录，靠 $ORIGIN 解析。
#
# 注意 $ORIGIN 的实测语义：glibc 按"打开这个对象时用的路径"展开 $ORIGIN，不是按
# 真实路径。所以从 torch/lib 的软链打开 libc10.so 时 $ORIGIN 是 torch/lib，找不到
# 同在 lib_dcu 的 libgflags-8aee0f6c.so.2.1.2。因此运行期预加载走 bundle 原路径
# （_vendor_libtorch._preload_global），不走软链 —— 那里有同样的说明。
#
# 不打包：DTK 驱动栈（libgalaxyhip.so.5 libMIOpen.so.1 librocblas.so.4
# libhipblas.so.2 librccl.so.1 等 12 个 soname）留在目标机 /opt/dtk。
# SDK 版本绑定本来就存在且更强（libtorch_hip.so 的 DT_NEEDED 写死
# librocblas.so.4），多打那 129 MB 不会让绑定更紧。
#
# 用法:
#   FLAGOS_DCU_TORCH_LIB=<dtk torch/lib> bash scripts/bundle_dcu_libtorch.sh
#   DTK_ROOT=/opt/dtk bash scripts/bundle_dcu_libtorch.sh
#
# 应在 `python setup.py bdist_wheel`（ACCELERATOR=dcu）之后、打 wheel 之前跑。幂等。

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/bundle_common.sh
source "${REPO_DIR}/scripts/lib/bundle_common.sh"

LIB_DCU="${REPO_DIR}/torch_fl/lib_dcu"
TORCH_FL_LIB="${REPO_DIR}/torch_fl/lib"
DTK_ROOT="${DTK_ROOT:-${ROCM_PATH:-/opt/dtk}}"

SRC="${FLAGOS_DCU_TORCH_LIB:-}"
if [ -z "${SRC}" ]; then
  SRC="$(bundle_find_vendor_torch_lib libtorch_hip.so dtk hip das || true)"
fi

if [ -z "${SRC}" ] || [ ! -d "${SRC}" ]; then
  echo "error: 找不到 DTK torch/lib。设 FLAGOS_DCU_TORCH_LIB=<dtk torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_hip.so" ]; then
  echo "error: ${SRC} 里没有 libtorch_hip.so，不是 DTK torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# 与 _dcu_libtorch_link._CORE_SO + _HIP_SO 对齐的自洽集合。
# libshm.so：libtorch_python.so 的直接 DT_NEEDED（torch.multiprocessing 的共享内存
# 管理器），DTK 的 torch/lib 里有实体文件，但它不在软链清单里，所以必须打进 bundle
# 让 $ORIGIN 找得到。
# libcaffe2_nvrtc.so：torch.cuda.init() 会 dlopen 它。DTK torch 带这个文件，
# stock +cpu wheel 没有，所以不打进来的话 GetFlagosDefaultCudaGenerator 里那次
# 按需 init 就死在 "Error in dlopen: libcaffe2_nvrtc.so"（RNG 修复反而暴露了它）。
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so libcaffe2_nvrtc.so)
HIP_SO=(libc10_hip.so libtorch_hip.so libmagma.so libshm.so)

# DTK 驱动栈实测分布（容器 LD_LIBRARY_PATH 与 find 结果一致）：
#   lib/          libhipnn librocfft.so.0 librocrand.so.1 librocsparse.so.1
#                 libMIOpen-recommend.so libunwind.so.8 libgalaxyhip.so.5
#   hip/lib/      hip runtime
#   aillvm/lib/   libomp.so
#   .hyhal/rocm_smi/lib/  librocm_smi64.so.2
# 全部留在目标机（装了 DCU 卡就有 /opt/dtk），但 RPATH 必须覆盖到，否则
# LD_LIBRARY_PATH 没设时就 not found。
VENDOR_RPATH="${DTK_ROOT}/lib:${DTK_ROOT}/hip/lib:${DTK_ROOT}/lib64"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/aillvm/lib:${DTK_ROOT}/.hyhal/rocm_smi/lib"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/llvm/lib:/opt/hyhal/lib"

echo "源 DTK torch/lib : ${SRC}"
echo "目标 lib_dcu     : ${LIB_DCU}"
echo "DTK 驱动路径     : ${DTK_ROOT}"

# bundle 内的库要能从两条路径被打开：
#   1. 直接从 lib_dcu/（运行期预加载走这条，见 _vendor_libtorch._preload_global）
#   2. 通过 torch/lib/ 的软链（`import torch` 自己加载 libtorch_global_deps.so 时）
# glibc 按"打开这个对象时用的路径"展开 $ORIGIN，所以第 2 条路上 $ORIGIN 是
# torch/lib，找不到同在 lib_dcu 的 libmpi-3fcb240d.so.40.40.3 等 auditwheel 改名库。
# 两个目录都在 site-packages 下同级（torch/lib -> ../../torch_fl/lib_dcu），
# 所以再加一条相对路径就同时覆盖两种情况。实测：不加这条，先 `import torch`
# 再 import torch_fl 会死在 libmpi not found。
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_dcu"

bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 1 "${CORE_SO[@]}"
bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${HIP_SO[@]}"

# torch.libs/：与 torch/ 同级的 auditwheel 目录，文件名带 hash 后缀，只能 glob。
TORCH_LIBS="$(cd "${SRC}/../.." && pwd)/torch.libs"
if [ -d "${TORCH_LIBS}" ]; then
  echo "源 torch.libs    : ${TORCH_LIBS}"
  _names=()
  while IFS= read -r f; do
    _names+=("$(basename "${f}")")
  done < <(find "${TORCH_LIBS}" -maxdepth 1 -type f -name '*.so*' | sort)
  if [ ${#_names[@]} -gt 0 ]; then
    bundle_copy_so "${TORCH_LIBS}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${_names[@]}"
  else
    echo "warning: ${TORCH_LIBS} 里没有 .so，跳过" >&2
  fi
else
  echo "warning: 找不到 ${TORCH_LIBS}；若 libtorch_cpu.so 的 DT_NEEDED 里有带 hash" >&2
  echo "         的通用库（libglog-*.so.0 等），目标机上会缺库。" >&2
fi

# libtorch_fl.so / libflagos.so 还需要 DTK 的 CUDA 兼容层 libcudart.so.12
# （cuda_runtime_compat，见 CMakeLists.txt 的 DCU_CUDA_ROOT）。cmake 已经把它写进
# RUNPATH 了，这里重写时不能丢 —— 否则干净环境里 libcudart.so.12 就 not found。
_DCU_CUDA_LIB64=""
for _c in "${DTK_ROOT}"/cuda/cuda-*/lib64; do
  if [ -f "${_c}/libcudart.so.12" ]; then
    _DCU_CUDA_LIB64="${_c}"
    break
  fi
done
if [ -z "${_DCU_CUDA_LIB64}" ]; then
  echo "warning: ${DTK_ROOT}/cuda/cuda-*/lib64 里没有 libcudart.so.12" >&2
fi
PLUGIN_RPATH="\$ORIGIN:\$ORIGIN/../lib_dcu"
[ -n "${_DCU_CUDA_LIB64}" ] && PLUGIN_RPATH="${PLUGIN_RPATH}:${_DCU_CUDA_LIB64}"
bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" "${PLUGIN_RPATH}:${VENDOR_RPATH}"

# torch/version.py 是构建期生成的纯 Python，换掉 .so 改不了它。自包含 DCU wheel
# 前面是 stock torch+cpu，torch.version.hip 报 None，而 triton 的 hcu backend
# 恰好按这个值判活（backends/hcu/driver.py is_active(): torch.cuda.is_available()
# and torch.version.hip is not None）。None 就永不激活，任何 flag_gems 算子都死在
# triton 的 driver factory："0 active drivers ([])"。把厂商 torch 自己的 version.py
# 带上，import 时由 _restore_dcu_hip_version() 读回 hip/rocm 字符串。
_VENDOR_VERSION_PY="$(cd "${SRC}/.." && pwd)/version.py"
if [ -f "${_VENDOR_VERSION_PY}" ]; then
  cp -fL "${_VENDOR_VERSION_PY}" "${LIB_DCU}/vendor_version.py"
  echo "已复制 vendor version.py -> lib_dcu/vendor_version.py"
  grep -E "^\s*(hip|rocm)\s*(:|=)" "${LIB_DCU}/vendor_version.py" || true
else
  echo "warning: 找不到 ${_VENDOR_VERSION_PY}，triton hcu backend 可能不激活" >&2
fi

bundle_summary "${LIB_DCU}"
bundle_check_needed "${LIB_DCU}" \
    "${DTK_ROOT}/lib" "${DTK_ROOT}/hip/lib" "${DTK_ROOT}/lib64" \
    "${DTK_ROOT}/aillvm/lib" "${DTK_ROOT}/llvm/lib" \
    "${DTK_ROOT}/.hyhal/rocm_smi/lib" "${_DCU_CUDA_LIB64:-/nonexistent}" \
    "/opt/hyhal/lib" "/usr/lib64" "/usr/lib" "/usr/lib/x86_64-linux-gnu"
