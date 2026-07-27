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

"""flagos NCCL backend extension.

Exposes c10d::ProcessGroupNCCL on CPU-only torch builds that ship without
USE_C10D_NCCL, by linking against an externally preloaded libtorch_cuda.so.
The compiled module (_flagos_nccl) is built out-of-tree via build.py; it may be
absent in slim installs, in which case ProcessGroupFlagOS falls back to FlagCX.
"""

try:
    from . import _flagos_nccl  # noqa: F401
except ImportError:  # pragma: no cover - extension not built
    _flagos_nccl = None
