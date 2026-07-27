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

import os

import pytest


def _detect_platform() -> str:
    """Infer the active hardware/backend platform from env."""
    accelerator = os.environ.get("ACCELERATOR", "").lower()
    if accelerator == "ascend":
        return "ascend"
    if accelerator in ("metax", "maca"):
        return "metax"

    backend_cfg = os.environ.get("FLAGOS_BACKEND_CONFIG", "").lower()
    if "ascend" in backend_cfg:
        return "ascend"
    if "metax" in backend_cfg:
        return "metax"
    return "default"


def _flaggems_enabled() -> bool:
    """Whether the FlagGems (Triton) path is actually active this run."""
    return os.environ.get("FLAGOS_USE_FLAGGEMS", "0").lower() not in (
        "0",
        "",
        "off",
        "false",
    )


# Markers to skip per platform (tests for other backends are not compiled/available).
# NOTE: `flaggems` (subprocess dispatch-log tests forcing a specific backend conf)
# stays platform-skipped on metax -- those assert the native CUDA flaggems dispatch
# path, not the metax boxing path. `flaggems_python` is instead gated at runtime by
# FLAGOS_USE_FLAGGEMS (see pytest_collection_modifyitems): it runs on top of the
# CUDA boxing path on any vendor, so it must run when FlagGems is enabled and skip
# otherwise, regardless of platform.
_PLATFORM_SKIP_MARKERS: dict[str, tuple[str, ...]] = {
    "metax": ("cuda", "ascend", "flaggems"),
    "ascend": ("cuda", "metax"),
    "default": ("metax", "ascend"),
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    platform = _detect_platform()
    markers_to_skip = list(_PLATFORM_SKIP_MARKERS.get(platform, ()))
    # In MetaX boxing mode the hand-written mxcc backend is NOT compiled: ops run
    # through the CUDA boxing kernels (and optionally the FlagGems Python path).
    # Tests asserting a `-> metax` dispatch (mark.metax) cannot pass, so skip them.
    if platform == "metax" and os.environ.get("FLAGOS_METAX_BOXING", "0") == "1":
        markers_to_skip.append("metax")
    # flaggems_python tests assert the FlagGems Triton path's runtime contract
    # (e.g. seedable/reproducible RNG). Without FLAGOS_USE_FLAGGEMS the same ops
    # route through the plain CUDA boxing kernels (native generator), where that
    # contract does not hold -> they must skip, not fail.
    if not _flaggems_enabled() and "flaggems_python" not in markers_to_skip:
        markers_to_skip.append("flaggems_python")
    for item in items:
        for marker_name in markers_to_skip:
            if item.get_closest_marker(marker_name):
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"Skipped on {platform} runtime: "
                            f"requires @{marker_name} backend"
                        )
                    )
                )
                break


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "metax: requires MetaX platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "flaggems: requires FlagGems (Triton) backend")
    config.addinivalue_line(
        "markers", "flaggems_python: requires FlagGems Python wrapper backend"
    )
