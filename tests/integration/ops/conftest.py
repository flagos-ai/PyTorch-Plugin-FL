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
    if not _is_metax_runtime():
        return

    skip_cuda_flaggems = pytest.mark.skip(
        reason="Skipped on metax runtime: test requires CUDA/flaggems backend"
    )
    keywords = (
        "dispatch_log_cuda",
        "flaggems",
        "flagos_default",
        "matches_cuda",
        "cuda_ref",
        "cuda_override",
    )
    for item in items:
        nodeid = item.nodeid
        if any(keyword in nodeid for keyword in keywords):
            item.add_marker(skip_cuda_flaggems)


def pytest_configure(config):
    config.addinivalue_line("markers", "anyplatform: runs on any platform")
    config.addinivalue_line("markers", "cuda: requires CUDA platform")
    config.addinivalue_line("markers", "ascend: requires Ascend platform")
    config.addinivalue_line("markers", "flaggems: requires FlagGems (Triton) backend")
    config.addinivalue_line(
        "markers", "flaggems_python: requires FlagGems Python wrapper backend"
    )
