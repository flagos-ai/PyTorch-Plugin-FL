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
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "ACCELERATOR=$ACCELERATOR" >> "$GITHUB_ENV"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is unavailable"
  exit 1
fi
nvidia-smi

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
PY
