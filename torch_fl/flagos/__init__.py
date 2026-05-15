import torch

from .. import _C  # type: ignore[misc]

from . import meta  # noqa: F401


_initialized = False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _C._exchangeDevice(self.idx)

    def __exit__(self, type, value, traceback):
        _C._set_device(self.prev_idx)
        return False


def is_available():
    return _C._get_device_count() > 0


def device_count() -> int:
    return _C._get_device_count()


def current_device():
    return _C._get_device()


def set_device(device) -> None:
    return _C._set_device(device)


def synchronize(device=None):
    r"""Waits for all operations on the flagos device to complete.

    Args:
        device (torch.device or int, optional): device to synchronize.
            It uses the current device, given by :func:`~torch_fl.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if device is not None:
        idx = torch.accelerator._get_device_index(device, optional=True)
        prev_idx = _C._exchangeDevice(idx)
        try:
            _C._synchronize()
        finally:
            _C._set_device(prev_idx)
    else:
        _C._synchronize()


def init():
    _lazy_init()


def is_initialized():
    return _initialized


def _lazy_init():
    global _initialized
    if is_initialized():
        return
    _C._init()
    _initialized = True


from .random import *  # noqa: F403, E402


# ---------------------------------------------------------------------------
# Stream API required by FSDP
# Since flagos shares the same GPU as CUDA, we proxy to torch.cuda streams.
# ---------------------------------------------------------------------------


class Stream(torch.cuda.Stream):
    """Flagos stream that wraps a CUDA stream (same GPU memory)."""

    def __new__(cls, device=None, priority=0, **kwargs):
        if device is None:
            device = current_device()
        return super().__new__(cls, device=device, priority=priority, **kwargs)


class Event(torch.cuda.Event):
    """Flagos event that wraps a CUDA event."""

    pass


def current_stream(device=None):
    """Return the currently selected stream for the given device."""
    if device is None:
        device = current_device()
    return torch.cuda.current_stream(device)


def stream(s):
    """Context-manager that selects a given stream."""
    return torch.cuda.stream(s)


def get_amp_supported_dtype():
    """Return list of supported dtypes for AMP (Automatic Mixed Precision).

    Required by torch.autocast for custom device backends.
    """
    return [torch.float16, torch.bfloat16]


__all__ = [
    "device",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "initial_seed",  # noqa: F405
    "is_available",
    "init",
    "is_initialized",
    "manual_seed",  # noqa: F405
    "manual_seed_all",  # noqa: F405
    "get_rng_state",  # noqa: F405
    "set_rng_state",  # noqa: F405
    "get_amp_supported_dtype",
    "Stream",
    "Event",
    "current_stream",
    "stream",
]
