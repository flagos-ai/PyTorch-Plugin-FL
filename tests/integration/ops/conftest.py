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


# Markers to skip per platform (tests for other backends are not compiled/available).
_PLATFORM_SKIP_MARKERS: dict[str, tuple[str, ...]] = {
    "metax": ("cuda", "ascend", "flaggems_python"),
    "ascend": ("cuda", "metax"),
    "default": ("metax", "ascend"),
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    platform = _detect_platform()
    markers_to_skip = _PLATFORM_SKIP_MARKERS.get(platform, ())
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
