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
FlagTree integration for torch.compile (Phase 2).

Patches inductor's Triton imports to use FlagTree instead of OpenAI Triton,
enabling multi-backend kernel compilation (NVIDIA/Ascend/Cambricon/MetaX).

Activated via FLAGOS_USE_FLAGTREE=1 environment variable.
"""

import sys


_original_triton = None
_patched = False


def patch_inductor_triton():
    """
    Replace inductor's triton imports with flagtree.

    FlagTree is API-compatible with OpenAI Triton (it's a fork), so this is
    a drop-in replacement. Inductor generates Triton kernel code; we just
    swap which compiler JITs it.

    This must be called before inductor imports triton (i.e., before the
    first torch.compile call that uses inductor).
    """
    global _patched, _original_triton

    if _patched:
        return

    try:
        import flagtree as triton
    except ImportError:
        raise ImportError(
            "FLAGOS_USE_FLAGTREE=1 but 'flagtree' package not installed. "
            "Install with: pip install flagtree"
        )

    # Save original triton if already imported
    if "triton" in sys.modules:
        _original_triton = sys.modules["triton"]

    # Replace triton in sys.modules with flagtree
    sys.modules["triton"] = triton

    # Also replace triton.language (inductor imports this)
    if hasattr(triton, "language"):
        sys.modules["triton.language"] = triton.language

    _patched = True


def unpatch_inductor_triton():
    """
    Restore original OpenAI Triton (for testing/cleanup).
    """
    global _patched, _original_triton

    if not _patched:
        return

    if _original_triton is not None:
        sys.modules["triton"] = _original_triton
        if hasattr(_original_triton, "language"):
            sys.modules["triton.language"] = _original_triton.language
    else:
        sys.modules.pop("triton", None)
        sys.modules.pop("triton.language", None)

    _patched = False
