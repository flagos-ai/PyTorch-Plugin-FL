#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

case "${CI_STAGE:-}" in
  build|integration) ;;
  *)
    echo "::error::CI_STAGE must be either 'build' or 'integration'"
    exit 1
    ;;
esac

export ACCELERATOR=cuda
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"

PYTHON_SITE=$(python - <<'PY'
import sysconfig

print(sysconfig.get_paths()["purelib"])
PY
)
TORCH_ROOT=$(python - <<'PY'
from pathlib import Path

import torch

print(Path(torch.__file__).resolve().parent)
PY
)

export FLAGGEMS_DIR="$PYTHON_SITE/flag_gems/lib/cmake/FlagGems"
export FLAGCX_PATH="${FLAGCX_PATH:-/opt/FlagCX}"
export CCL_HOME="$PYTHON_SITE/nvidia/nccl"

export CMAKE_PREFIX_PATH="$FLAGGEMS_DIR:$TORCH_ROOT/share/cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$CUDA_HOME/include:$CCL_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/lib64:$CCL_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TORCH_ROOT/lib:$PYTHON_SITE/flag_gems/lib:$CCL_HOME/lib:$FLAGCX_PATH/build/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  for name in \
    ACCELERATOR CUDA_HOME CUDA_PATH FLAGGEMS_DIR FLAGCX_PATH CCL_HOME \
    CMAKE_PREFIX_PATH CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
    printf '%s=%s\n' "$name" "${!name}" >> "$GITHUB_ENV"
  done
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
from importlib.metadata import version
from pathlib import Path

import flag_gems
import flagcx
import torch

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
assert torch.cuda.device_count() > 0, "No CUDA devices were detected"
assert torch.__version__.startswith("2.10.0+cu130"), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert version("flag_gems").startswith("5.3"), version("flag_gems")
assert version("flagcx").startswith("0.13.0"), version("flagcx")

torch_root = Path(torch.__file__).resolve().parent
gems_root = Path(flag_gems.__file__).resolve().parent
assert (torch_root / "lib/libtorch.so").is_file(), "libtorch.so is missing"
assert list(gems_root.rglob("FlagGemsConfig.cmake")), "FlagGemsConfig.cmake is missing"
assert list(gems_root.rglob("liboperators.so")), "liboperators.so is missing"

print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"FlagGems: {version('flag_gems')}; FlagCX: {version('flagcx')}")
PY
