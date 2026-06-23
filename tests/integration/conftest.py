import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_backend_config() -> None:
    """Ensure MetaX backend config is set before importing torch_fl (if not already specified)."""
    if os.environ.get("FLAGOS_BACKEND_CONFIG"):
        return
    accel = os.environ.get("ACCELERATOR", "").lower()
    use_metax = accel in ("metax", "maca") or Path("/dev/mxcd").exists()
    if use_metax:
        cfg = _REPO_ROOT / "torch_fl" / "backends_metax.conf"
        if cfg.is_file():
            os.environ["FLAGOS_BACKEND_CONFIG"] = str(cfg)


_ensure_backend_config()


def pytest_addoption(parser):
    parser.addoption(
        "--model", default="Qwen/Qwen3-0.6B", help="Path to Qwen3 model"
    )
    parser.addoption(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max new tokens for inference tests",
    )
    parser.addoption("--steps", type=int, default=10, help="Training steps")
    parser.addoption(
        "--batch-size", type=int, default=2, help="Batch size for training"
    )
    parser.addoption(
        "--seq-len", type=int, default=1024, help="Sequence length for training"
    )
    parser.addoption(
        "--lr", type=float, default=1e-5, help="Learning rate for training"
    )


def pytest_configure(config):
    import torch_fl  # noqa: F401

    if not torch_fl.flagos.is_available():
        pytest.exit("flagos device is not available.")

    # Initialize flagos device to ensure ACL runtime is properly set up
    torch_fl.flagos.init()
