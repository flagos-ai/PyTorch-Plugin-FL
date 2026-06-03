import os

import pytest


def _is_metax_runtime() -> bool:
    accelerator = os.environ.get("ACCELERATOR", "").lower()
    if accelerator == "metax":
        return True

    backend_cfg = os.environ.get("FLAGOS_BACKEND_CONFIG", "").lower()
    return "metax" in backend_cfg


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _is_metax_runtime():
        skip_cuda_flaggems = pytest.mark.skip(
            reason="Skipped on metax runtime: test requires CUDA/flaggems backend"
        )
        cuda_keywords = (
            "dispatch_log_cuda",
            "flaggems",
            "flagos_default",
            "matches_cuda",
            "cuda_ref",
            "cuda_override",
        )
        for item in items:
            nodeid = item.nodeid
            if any(keyword in nodeid for keyword in cuda_keywords):
                item.add_marker(skip_cuda_flaggems)
    else:
        skip_metax = pytest.mark.skip(
            reason="Skipped off metax runtime: test requires MetaX backend"
        )
        for item in items:
            if item.get_closest_marker("metax") or "dispatch_log_metax" in item.nodeid:
                item.add_marker(skip_metax)


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "metax: requires MetaX platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "flaggems: requires FlagGems (Triton) backend")
    config.addinivalue_line(
        "markers", "flaggems_python: requires FlagGems Python wrapper backend"
    )
