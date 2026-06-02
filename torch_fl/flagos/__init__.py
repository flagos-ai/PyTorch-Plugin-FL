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

    # Eagerly import FlagGems to avoid deep import chain during dispatch.
    # FlagGems has a deep lazy import chain (fused → FLA → utils → models → sqlalchemy)
    # that can exceed Python's recursion limit when triggered inside PyTorch dispatch.
    try:
        import flag_gems  # noqa: F401
    except ImportError:
        pass  # FlagGems not installed, skip

    # Monkey-patch Tensor.__getitem__ to work around PyTorch C++ dispatch issue
    # with advanced indexing on custom devices. The C++ __getitem__ fails for
    # patterns like x[:, tensor_idx] but torch.ops.aten.index.Tensor works.
    import torch

    _original_getitem = torch.Tensor.__getitem__

    def _patched_getitem(self, indices):
        # Only patch for our device
        if self.device.type not in ("privateuseone", "flagos"):
            return _original_getitem(self, indices)

        # Handle tuple of indices with at least one tensor
        if isinstance(indices, tuple):
            has_tensor = any(isinstance(idx, torch.Tensor) for idx in indices)
            if has_tensor:
                # Convert to list for aten.index.Tensor
                indices_list = []
                for idx in indices:
                    if isinstance(idx, slice):
                        if idx == slice(None, None, None):
                            indices_list.append(None)
                        else:
                            # Non-trivial slice — fall back to original
                            return _original_getitem(self, indices)
                    elif isinstance(idx, torch.Tensor):
                        indices_list.append(idx)
                    else:
                        # Other types (int, etc.) — fall back to original
                        return _original_getitem(self, indices)

                # Use aten.index.Tensor which works correctly
                return torch.ops.aten.index.Tensor(self, indices_list)

        return _original_getitem(self, indices)

    torch.Tensor.__getitem__ = _patched_getitem


from .random import *  # noqa: F403, E402


# default_generators: list of one Generator per device, required by FlagGems
class _DefaultGenerators:
    """Lazy list-like accessor for per-device default generators."""

    def __getitem__(self, device):
        return _C._get_default_generator(device)

    def __len__(self):
        return device_count()


default_generators = _DefaultGenerators()


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


class Event:
    """Simple timing event using host-side timestamps after device sync."""

    def __init__(
        self, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        self._time = None

    def record(self, stream=None):
        import time as _time

        _C._synchronize()
        self._time = _time.perf_counter()

    def elapsed_time(self, end_event):
        if self._time is None or end_event._time is None:
            raise RuntimeError("Events have not been recorded")
        return (end_event._time - self._time) * 1000.0

    def synchronize(self):
        _C._synchronize()

    def query(self):
        return True

    def wait(self, stream=None):
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
    "default_generators",
]
