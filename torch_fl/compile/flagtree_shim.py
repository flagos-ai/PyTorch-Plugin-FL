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

"""
FlagTree detection for torch.compile.

FlagTree is a Triton fork that substitutes itself for Triton *at install time*:
its wheel is named ``flagtree``, but the module it installs is ``triton``, and
installing it uninstalls the official ``triton``. So inductor's own
``import triton`` already resolves to FlagTree once it is installed, and nothing
here needs to patch ``sys.modules`` -- there is no ``flagtree`` module to import.

This module therefore only *reports* which Triton is active, so that
FLAGOS_USE_FLAGTREE=1 can assert FlagTree is really in use instead of silently
compiling with stock Triton.
"""

import importlib.util
from typing import Optional


# FlagTree-only module. Present regardless of which vendor backend was built,
# unlike triton._flagtree_backend.FLAGTREE_BACKEND, which is the empty string on
# nvidia/amd because upstream tells you not to set FLAGTREE_BACKEND for those.
_FLAGTREE_MARKER = "triton._flagtree_spec"


def is_flagtree_active() -> bool:
    """Whether the importable ``triton`` is FlagTree rather than stock Triton."""
    try:
        return importlib.util.find_spec(_FLAGTREE_MARKER) is not None
    except (ImportError, ValueError):
        # No triton at all, or a triton too broken to introspect.
        return False


def flagtree_backend() -> Optional[str]:
    """The vendor backend FlagTree was built for, if it recorded one.

    Empty on nvidia/amd (upstream builds those without FLAGTREE_BACKEND set), so
    an empty result means "FlagTree, vendor unrecorded", not "not FlagTree".
    Returns None when FlagTree is not active.
    """
    if not is_flagtree_active():
        return None
    try:
        from triton._flagtree_backend import FLAGTREE_BACKEND

        return FLAGTREE_BACKEND
    except ImportError:
        return ""


def require_flagtree() -> None:
    """Fail unless the active Triton is FlagTree.

    Called for FLAGOS_USE_FLAGTREE=1. FlagTree cannot be enabled from here -- it
    is chosen when the environment is built -- so the only useful thing this can
    do is refuse to pretend.
    """
    if is_flagtree_active():
        return

    raise RuntimeError(
        "FLAGOS_USE_FLAGTREE=1 but the active 'triton' module is stock Triton, "
        "not FlagTree. FlagTree replaces triton at install time; it is not "
        "something this process can switch on. Build it from source "
        "(https://github.com/flagos-ai/FlagTree) -- there is no 'flagtree' "
        "package on PyPI, and 'import flagtree' never works because the module "
        "it installs is named 'triton'. Unset FLAGOS_USE_FLAGTREE to compile "
        "with stock Triton instead."
    )
