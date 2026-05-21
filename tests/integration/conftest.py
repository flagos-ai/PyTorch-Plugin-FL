import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--model", default="/nfs/hcr/models/Qwen/Qwen3-0.6B", help="Path to Qwen3 model"
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
