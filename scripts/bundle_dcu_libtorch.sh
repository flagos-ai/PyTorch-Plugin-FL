#!/usr/bin/env bash
# Bundle the DTK-forked libtorch C++ .so into torch_fl/lib_dcu/ for a
# self-contained single wheel.
#
# Why bundle the core libs (not just libtorch_hip.so):
# Measured symbol attribution: DTK's libtorch_cpu.so resolves 2101 undefined
# symbols in libtorch_fl.so, while libtorch_hip.so resolves 0 -- the fork is on
# the core-lib side (DTK's libtorch_cpu.so exports 128 hip symbols and carries
# DT_NEEDED: libgalaxyhip.so.5). libtorch_hip.so still has to be present: the
# boxing kernels require vendor kernels to be registered under the CUDA dispatch
# key.
#
# Why bundle torch.libs/: those 17 files are auditwheel-mangled common libs
# (glog/gflags/MKL/OpenMPI/hwloc/libxml2...), 5 of which are direct DT_NEEDED of
# DTK's libtorch_cpu.so. Measured, they carry 0 hip/dtk/rocm references and
# introduce no SDK binding; but the official torch+cpu wheel's torch.libs
# contains none of them (names carry a hash, system libs do not match either),
# so the wheel cannot run without bundling them. They sit in the same dir as the
# core libs and are resolved via $ORIGIN.
#
# Note on $ORIGIN semantics: glibc expands $ORIGIN from the path the object was
# *opened by*, not from the resolved real path. So opening libc10.so through the
# torch/lib symlink gives $ORIGIN = torch/lib, and it cannot find
# libgflags-8aee0f6c.so.2.1.2 that sits alongside it in lib_dcu. Runtime
# preloading therefore walks the bundle's original paths
# (_vendor_libtorch._preload_global), not the symlinks -- the same rationale is
# documented there.
#
# Not bundled: DTK driver stack (libgalaxyhip.so.5 libMIOpen.so.1
# librocblas.so.4 libhipblas.so.2 librccl.so.1, etc., 12 sonames) stays on the
# target machine under /opt/dtk. SDK version binding already exists and is
# stronger (libtorch_hip.so's DT_NEEDED hard-codes librocblas.so.4); bundling
# that extra 129 MB would not tighten the binding.
#
# Usage:
#   FLAGOS_DCU_TORCH_LIB=<dtk torch/lib> bash scripts/bundle_dcu_libtorch.sh
#   DTK_ROOT=/opt/dtk bash scripts/bundle_dcu_libtorch.sh
#
# Should run after `python setup.py bdist_wheel` (ACCELERATOR=dcu) and before
# packing the wheel. Idempotent.

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
  echo "error: DTK torch/lib not found. Set FLAGOS_DCU_TORCH_LIB=<dtk torch/lib>" >&2
  exit 1
fi
if [ ! -f "${SRC}/libtorch_hip.so" ]; then
  echo "error: ${SRC} does not contain libtorch_hip.so, not a DTK torch/lib" >&2
  exit 1
fi

bundle_require_patchelf

# Self-consistent set aligned with _dcu_libtorch_link._CORE_SO + _HIP_SO.
# libshm.so: direct DT_NEEDED of libtorch_python.so (torch.multiprocessing's
# shared-memory manager); DTK's torch/lib has the real file, but it is not on
# the symlink manifest, so it must be bundled for $ORIGIN resolution.
# libcaffe2_nvrtc.so: torch.cuda.init() dlopens it. DTK torch ships this file,
# stock +cpu wheel does not, so without bundling it the on-demand init inside
# GetFlagosDefaultCudaGenerator dies on "Error in dlopen: libcaffe2_nvrtc.so"
# (the RNG fix exposed it).
CORE_SO=(libc10.so libtorch_cpu.so libtorch.so libtorch_global_deps.so libtorch_python.so libcaffe2_nvrtc.so)
HIP_SO=(libc10_hip.so libtorch_hip.so libmagma.so libshm.so)

# DTK driver stack measured layout (container LD_LIBRARY_PATH matches find results):
#   lib/          libhipnn librocfft.so.0 librocrand.so.1 librocsparse.so.1
#                 libMIOpen-recommend.so libunwind.so.8 libgalaxyhip.so.5
#   hip/lib/      hip runtime
#   aillvm/lib/   libomp.so
#   .hyhal/rocm_smi/lib/  librocm_smi64.so.2
# All stay on the target machine (a box with DCU cards has /opt/dtk), but RPATH
# must cover them, else not-found when LD_LIBRARY_PATH is unset.
VENDOR_RPATH="${DTK_ROOT}/lib:${DTK_ROOT}/hip/lib:${DTK_ROOT}/lib64"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/aillvm/lib:${DTK_ROOT}/.hyhal/rocm_smi/lib"
VENDOR_RPATH="${VENDOR_RPATH}:${DTK_ROOT}/llvm/lib:/opt/hyhal/lib"

echo "Source DTK torch/lib : ${SRC}"
echo "Target lib_dcu       : ${LIB_DCU}"
echo "DTK driver path      : ${DTK_ROOT}"

# Libs inside the bundle must be openable from two paths:
#   1. Directly from lib_dcu/ (runtime preload walks this one, see
#      _vendor_libtorch._preload_global)
#   2. Through the torch/lib/ symlinks (when `import torch` loads
#      libtorch_global_deps.so itself)
# glibc expands $ORIGIN from the path the object was opened by, so path #2 gives
# $ORIGIN = torch/lib and cannot find libmpi-3fcb240d.so.40.40.3 and other
# auditwheel-mangled libs sitting in lib_dcu. Both dirs are siblings under
# site-packages (torch/lib -> ../../torch_fl/lib_dcu), so adding one more
# relative path covers both cases. Measured: without this, `import torch` before
# `import torch_fl` dies on libmpi not found.
BUNDLE_ORIGIN="\$ORIGIN:\$ORIGIN/../../torch_fl/lib_dcu"

bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 1 "${CORE_SO[@]}"
bundle_copy_so "${SRC}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${HIP_SO[@]}"

# torch.libs/: auditwheel dir sibling to torch/, filenames carry hash suffixes,
# can only glob.
TORCH_LIBS="$(cd "${SRC}/../.." && pwd)/torch.libs"
if [ -d "${TORCH_LIBS}" ]; then
  echo "Source torch.libs    : ${TORCH_LIBS}"
  _names=()
  while IFS= read -r f; do
    _names+=("$(basename "${f}")")
  done < <(find "${TORCH_LIBS}" -maxdepth 1 -type f -name '*.so*' | sort)
  if [ ${#_names[@]} -gt 0 ]; then
    bundle_copy_so "${TORCH_LIBS}" "${LIB_DCU}" "${BUNDLE_ORIGIN}:${VENDOR_RPATH}" 0 "${_names[@]}"
  else
    echo "warning: ${TORCH_LIBS} has no .so, skipping" >&2
  fi
else
  echo "warning: ${TORCH_LIBS} not found; the target will lack hash-suffixed" >&2
  echo "         common libs (libglog-*.so.0 etc.) from libtorch_cpu.so's DT_NEEDED." >&2
fi

# libtorch_fl.so / libflagos.so also need DTK's CUDA compatibility layer
# libcudart.so.12 (cuda_runtime_compat, see CMakeLists.txt's DCU_CUDA_ROOT).
# cmake already wrote it into RUNPATH; cannot drop it when rewriting here --
# else libcudart.so.12 becomes not-found in a clean environment.
_DCU_CUDA_LIB64=""
for _c in "${DTK_ROOT}"/cuda/cuda-*/lib64; do
  if [ -f "${_c}/libcudart.so.12" ]; then
    _DCU_CUDA_LIB64="${_c}"
    break
  fi
done
if [ -z "${_DCU_CUDA_LIB64}" ]; then
  echo "warning: ${DTK_ROOT}/cuda/cuda-*/lib64 has no libcudart.so.12" >&2
fi
PLUGIN_RPATH="\$ORIGIN:\$ORIGIN/../lib_dcu"
[ -n "${_DCU_CUDA_LIB64}" ] && PLUGIN_RPATH="${PLUGIN_RPATH}:${_DCU_CUDA_LIB64}"
bundle_rewrite_plugin_rpath "${TORCH_FL_LIB}" "${PLUGIN_RPATH}:${VENDOR_RPATH}"

# torch/version.py is pure Python generated at build time, so swapping .so
# cannot change it. In the self-contained DCU wheel the base is stock torch+cpu,
# whose torch.version.hip reports None, and triton's hcu backend gates
# is_active() on that value (backends/hcu/driver.py is_active():
# torch.cuda.is_available() and torch.version.hip is not None). None means never
# activate, and any flag_gems op dies in triton's driver factory: "0 active
# drivers ([])". Carry the vendor torch's own version.py; at import time
# _restore_dcu_hip_version() reads back the hip/rocm strings.
_VENDOR_VERSION_PY="$(cd "${SRC}/.." && pwd)/version.py"
if [ -f "${_VENDOR_VERSION_PY}" ]; then
  cp -fL "${_VENDOR_VERSION_PY}" "${LIB_DCU}/vendor_version.py"
  echo "Copied vendor version.py -> lib_dcu/vendor_version.py"
  grep -E "^\s*(hip|rocm)\s*(:|=)" "${LIB_DCU}/vendor_version.py" || true
else
  echo "warning: ${_VENDOR_VERSION_PY} not found, triton hcu backend may not activate" >&2
fi

bundle_summary "${LIB_DCU}"
bundle_check_needed "${LIB_DCU}" \
    "${DTK_ROOT}/lib" "${DTK_ROOT}/hip/lib" "${DTK_ROOT}/lib64" \
    "${DTK_ROOT}/aillvm/lib" "${DTK_ROOT}/llvm/lib" \
    "${DTK_ROOT}/.hyhal/rocm_smi/lib" "${_DCU_CUDA_LIB64:-/nonexistent}" \
    "/opt/hyhal/lib" "/usr/lib64" "/usr/lib" "/usr/lib/x86_64-linux-gnu"
