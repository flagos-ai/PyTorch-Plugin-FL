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

"""Build the flagos NCCL backend extension (_flagos_nccl).

Only needed on a CPU-only torch build where torch.distributed does not expose
ProcessGroupNCCL, but an external libtorch_cuda.so (containing the NCCL backend)
is available. See nccl_backend.cpp for the rationale.

Usage (from repo root, with the external cuda assets present):

    SP=<env>/site-packages
    export LIBRARY_PATH=".libtorch_cuda_assets:$SP/torch/lib:$(ls -d $SP/nvidia/*/lib|tr '\\n' ':')"
    export LD_LIBRARY_PATH="$LIBRARY_PATH"
    python torch_fl/comm/_nccl_ext/build.py

Produces _flagos_nccl*.so next to this file.
"""

import glob
import os
import sys

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))


def _library_dirs():
    dirs = []
    # External cuda assets: libc10_cuda.so / libtorch_cuda.so live here.
    assets = os.path.join(_REPO, ".libtorch_cuda_assets")
    if os.path.isdir(assets):
        dirs.append(assets)
    # Also the installed torch_fl/lib (build output) as a fallback.
    tf_lib = os.path.join(_REPO, "torch_fl", "lib")
    if os.path.isdir(tf_lib):
        dirs.append(tf_lib)
    # pip nvidia-*-cu12 wheels provide libnccl.so etc.
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is not None and spec.submodule_search_locations:
        for base in spec.submodule_search_locations:
            dirs.extend(sorted(glob.glob(os.path.join(base, "*", "lib"))))
    return dirs


def main():
    lib_dirs = _library_dirs()
    ext = CppExtension(
        name="_flagos_nccl",
        sources=[os.path.join(_HERE, "nccl_backend.cpp")],
        # -DUSE_C10D_NCCL unlocks the ProcessGroupNCCL.hpp header, which was
        # #ifdef'd out of the CPU wheel. The NCCL_HAS_* feature macros are then
        # derived from the pip nccl.h (2.28) to match the external .so ABI.
        define_macros=[("USE_C10D_NCCL", None),
                       ("C10_CUDA_NO_CMAKE_CONFIGURE_FILE", None)],
        include_dirs=[
            # CUDA toolkit headers (cuda.h, pulled in by ATen CUDA headers).
            os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"),
                         "include"),
            # pip nccl header (nccl.h): <base>/nvidia/nccl/lib -> .../include
            *[d[: -len("/lib")] + "/include"
              for d in lib_dirs if d.endswith("nccl/lib")],
        ],
        library_dirs=lib_dirs,
        # c10_cuda / torch_cuda come from the external assets; nccl from pip.
        libraries=["c10_cuda", "torch_cuda", "nccl"],
        extra_compile_args=["-std=c++17"],
        # rpath so the .so finds the external libs at runtime too.
        extra_link_args=[f"-Wl,-rpath,{d}" for d in lib_dirs],
    )
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(
        name="_flagos_nccl",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    main()
