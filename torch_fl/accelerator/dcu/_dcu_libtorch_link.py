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

"""Symlink DTK's libtorch .so into the active (official) torch wheel's lib dir.

DCU runs the CUDA boxing kernels on top of DTK's hipified libtorch: HIP kernels
are registered under the CUDA dispatch key, so PrivateUse1 -> CUDA re-dispatch
works unchanged.  That libtorch is a fork -- measured, DTK's ``libtorch_cpu.so``
exports 128 hip symbols and carries ``DT_NEEDED: libgalaxyhip.so.5`` -- and it
resolves 2101 of ``libtorch_fl.so``'s undefined symbols, while ``libtorch_hip.so``
resolves 0.  So the core libs are what has to be swapped in; ``libtorch_hip.so``
still has to be *present* for the dispatch-key registration to exist.

See ``torch_fl.accelerator._vendor_libtorch`` for the symlink mechanism and why a
ctypes preload cannot replace core libs.

Unlike MetaX there is no env gate: a DCU wheel is only ever installed on a DCU
box, and the relink is skipped automatically when ``torch_fl/lib_dcu/`` was not
bundled (a plain in-place build) or when torch already IS the DTK wheel.

The DTK driver stack itself (``libgalaxyhip.so.5``, ``libMIOpen.so.1``,
``librocblas.so.4``, ``librccl.so.1``, ...) stays on the target under
``/opt/dtk`` and is reached via the RUNPATH baked in by
``cmake/FlagosRpath.cmake`` / ``scripts/bundle_dcu_libtorch.sh``.
"""

from torch_fl.accelerator._vendor_libtorch import (
    bundled_lib_dir,
    discover_vendor_torch_lib,
    ensure_vendor_libtorch_links,
)
from torch_fl.accelerator._vendor_libtorch import (
    restore_original_libtorch as _restore,
)

_BUNDLE_DIR = "lib_dcu"

# Core C++ .so that must come from the DTK wheel as a self-consistent set.
_CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
# HIP-side .so the stock +cpu wheel does not ship at all. libmagma.so is a
# DT_NEEDED of DTK's libtorch_hip.so; libshm.so is one of libtorch_python.so's
# (torch.multiprocessing's shared-memory manager) and is a *different* build in
# the DTK wheel, so it has to come from the same set.
_HIP_SO = (
    "libc10_hip.so",
    "libtorch_hip.so",
    "libmagma.so",
    "libshm.so",
)

# Dependency order for the RTLD_GLOBAL preload: core (CPU) first, then the HIP
# side, then libshm and libtorch_python. Same reasoning as MetaX -- symbols the
# plugin needs live in the forked CPU runtime, and loading only the HIP lib can
# leave its CPU dependency RTLD_LOCAL -- including why libshm.so has to be listed
# explicitly ahead of libtorch_python.so.
_LOAD_ORDER = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libc10_hip.so",
    "libtorch_hip.so",
    "libshm.so",
    "libtorch_python.so",
)

_MARKERS = ("dtk", "hip", "das")


def _bundled_dcu_lib():
    """Forked libtorch bundled inside this wheel (self-contained DCU build)."""
    return bundled_lib_dir(_BUNDLE_DIR, "libtorch_hip.so")


def _discover_dcu_torch_lib():
    """Locate the DTK libtorch .so dir.

    Priority: bundled lib_dcu/, then FLAGOS_DCU_TORCH_LIB, then sibling conda
    envs whose torch is a DTK build.
    """
    return discover_vendor_torch_lib(
        _BUNDLE_DIR,
        "libtorch_hip.so",
        env_override="FLAGOS_DCU_TORCH_LIB",
        vendor_markers=_MARKERS,
    )


def ensure_dcu_libtorch_links():
    """Symlink the active torch wheel's core .so to DTK's copies.

    Idempotent; reversible via ``torch/lib/_orig_backup/``.  Returns True if
    links are in place (or already were), False if there was nothing to do.
    """
    return ensure_vendor_libtorch_links(
        _BUNDLE_DIR,
        _CORE_SO,
        extra_so=_HIP_SO,
        env_override="FLAGOS_DCU_TORCH_LIB",
        vendor_markers=_MARKERS,
        probe_so="libtorch_hip.so",
        vendor="DCU/DTK",
        load_order=_LOAD_ORDER,
    )


def restore_original_libtorch():
    """Undo ensure_dcu_libtorch_links(): remove links, restore backups."""
    _restore(_CORE_SO, _HIP_SO, bundle_dirname=_BUNDLE_DIR)
