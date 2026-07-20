#!/usr/bin/env bash
# Build + run the aclnn codegen feasibility prototype (docs/ascend_aclnn_codegen_prototype.cc).
# See docs/ascend_npu_plan.md §可行性验证.
set -e

AH="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_DIR}/docs/ascend_aclnn_codegen_prototype.cc"
BIN="/tmp/ascend_aclnn_prototype"

g++ -std=c++17 "$SRC" -o "$BIN" \
  -I"${AH}/include" \
  -L"${AH}/lib64" \
  -lascendcl -lopapi -lnnopbase

echo "built -> $BIN"
LD_LIBRARY_PATH="${AH}/lib64:${LD_LIBRARY_PATH}" "$BIN"
