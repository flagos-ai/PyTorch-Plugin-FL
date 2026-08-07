#!/usr/bin/env bash
# 自包含 wheel 打包的公共部分，被 scripts/bundle_{maca,dcu,ppu}_libtorch.sh source。
#
# 背景：metax / dcu / ppu 三个后端跑的都是厂商 fork 的 libtorch —— 实测
# libtorch_cpu.so 提供 libtorch_fl.so 两千多个未定义符号，而厂商库
# （libtorch_hip.so / libtorch_cuda.so）提供 0 个。所以自包含 wheel 必须把 core
# 库一起打进来，不能只打厂商库。做法统一为：
#   - bundle 目录内的 .so         -> RPATH = $ORIGIN:<厂商驱动目录...>
#     （驱动运行时留在目标机，不打包：装了卡的机器必有驱动）
#   - torch_fl/lib/libtorch_fl.so -> RPATH = $ORIGIN:$ORIGIN/../<bundle>:<驱动...>
#     （去掉构建机写死的绝对路径。cmake/FlagosRpath.cmake 已经这么设了，
#      这里的 patchelf 是对已有 .so / 非 cmake 构建产物的兜底，保持幂等。）
#
# 运行期把 bundle 里的 core 库软链到 stock torch/lib 的逻辑在
# torch_fl/accelerator/_vendor_libtorch.py，两边的 .so 清单必须对齐。
#
# 提供:
#   bundle_require_patchelf
#   bundle_find_vendor_torch_lib <probe_so> <marker...>
#   bundle_copy_so <src_dir> <dst_dir> <rpath> <required:0|1> <so...>
#   bundle_rewrite_plugin_rpath <torch_fl_lib> <rpath>
#   bundle_summary <dst_dir>

set -euo pipefail

# patchelf 四个节点默认都没有。bw1000/810e 上 `pip install patchelf` 直接可用；
# mc550 的默认 index 没有这个包，要指定源。
bundle_require_patchelf() {
  if command -v patchelf >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<'EOF'
error: 找不到 patchelf，无法重写 RPATH。装法（任选）:
    pip install patchelf
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple patchelf   # 默认源没有时
    conda install -c conda-forge patchelf
EOF
  return 1
}

# 定位厂商 torch/lib：优先当前解释器的 torch（校验 version.py 含厂商标记且
# probe_so 存在），失败返回非 0。调用方应先看自己的 env 覆盖变量。
# 用法: bundle_find_vendor_torch_lib libtorch_hip.so dtk hip
bundle_find_vendor_torch_lib() {
  local probe="$1"
  shift
  local markers="$*"
  local py
  py="$(command -v python || command -v python3 || true)"
  [ -n "${py}" ] || return 1
  PROBE_SO="${probe}" VENDOR_MARKERS="${markers}" "${py}" - <<'PY' 2>/dev/null || return 1
import importlib.util, os

probe = os.environ["PROBE_SO"]
markers = os.environ["VENDOR_MARKERS"].split()
spec = importlib.util.find_spec("torch")
if spec and spec.submodule_search_locations:
    root = spec.submodule_search_locations[0]
    lib = os.path.join(root, "lib")
    ver = os.path.join(root, "version.py")
    txt = open(ver).read() if os.path.isfile(ver) else ""
    if (not markers or any(m in txt for m in markers)) and os.path.exists(
        os.path.join(lib, probe)
    ):
        print(lib)
PY
}

# 拷 .so 并设 RPATH。cp -fL 解引用软链，拷实体文件。
# required=1 时源缺失即失败，=0 时跳过（stock +cpu wheel 本来就没有厂商库）。
bundle_copy_so() {
  local src_dir="$1" dst_dir="$2" rpath="$3" required="$4"
  shift 4
  local so src
  mkdir -p "${dst_dir}"
  for so in "$@"; do
    src="${src_dir}/${so}"
    if [ ! -f "${src}" ]; then
      if [ "${required}" = "1" ]; then
        echo "error: 缺少必需的 ${src}" >&2
        return 1
      fi
      echo "  跳过 ${so}（源不存在）"
      continue
    fi
    cp -fL "${src}" "${dst_dir}/${so}"
    # 非 ELF（极少数 .so 其实是 linker script）patchelf 会失败，不致命。
    if ! patchelf --set-rpath "${rpath}" "${dst_dir}/${so}" 2>/dev/null; then
      echo "  ${so}: patchelf 跳过（非 ELF?）"
    fi
    echo "  打包 ${so} ($(du -h "${dst_dir}/${so}" | cut -f1))"
  done
}

# 重写 libtorch_fl.so / libtorch_bindings.so / libflagos.so 的 RPATH。
bundle_rewrite_plugin_rpath() {
  local lib_dir="$1" rpath="$2" so target
  for so in libtorch_fl.so libtorch_bindings.so libflagos.so; do
    target="${lib_dir}/${so}"
    [ -f "${target}" ] || continue
    patchelf --set-rpath "${rpath}" "${target}"
    echo "  重写 RPATH ${so} -> ${rpath}"
  done
}

bundle_summary() {
  local dst_dir="$1"
  echo "完成。$(basename "${dst_dir}") 总大小: $(du -sh "${dst_dir}" | cut -f1)（$(find "${dst_dir}" -type f | wc -l | tr -d ' ') 个文件）"
}

# 目标机上一定有的基线库，自检时不报。
_BUNDLE_BASELINE_RE='^(libc|libm|libdl|librt|libpthread|libstdc\+\+|libgcc_s|ld-linux.*|libutil|libresolv|libnsl|libcrypt|libatomic|libgomp|libz|libnuma)\.so'

# 自检：bundle 里每个 .so 的 DT_NEEDED，凡是既不在 bundle 内、也不在给定的目标机
# 目录里、又不是基线系统库的，都报出来 —— 那就是目标机上会 "not found" 的候选。
# 用法: bundle_check_needed <bundle_dir> <target_dir...>
bundle_check_needed() {
  local dst_dir="$1"
  shift
  local so dep d found
  echo "---- DT_NEEDED 自检（只报可能在目标机缺失的项）----"
  local missing=0
  for so in "${dst_dir}"/*.so*; do
    [ -f "${so}" ] || continue
    while read -r dep; do
      [ -n "${dep}" ] || continue
      [[ "${dep}" =~ ${_BUNDLE_BASELINE_RE} ]] && continue
      [ -e "${dst_dir}/${dep}" ] && continue
      found=0
      for d in "$@"; do
        if [ -e "${d}/${dep}" ]; then found=1; break; fi
      done
      [ "${found}" = "1" ] && continue
      echo "  $(basename "${so}") -> ${dep}"
      missing=1
    done < <(patchelf --print-needed "${so}" 2>/dev/null || true)
  done
  if [ "${missing}" = "0" ]; then
    echo "  无（bundle + 驱动目录已覆盖全部非基线依赖）"
  fi
}
