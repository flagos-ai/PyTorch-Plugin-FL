"""
Integration module for registering FlagGems operators with the flagos device.

This module provides utilities to register FlagGems' Triton kernels as
implementations for the PrivateUse1 (flagos) dispatch key.

FlagGems operators are automatically registered when torch_fl is imported.
You can use the functions in this module for more fine-grained control.
"""

import torch
from typing import Optional, List, Callable, Set

# Track registered operators
_registered_ops: Set[str] = set()


def get_registered_ops() -> List[str]:
    """Return list of currently registered FlagGems operators."""
    return list(_registered_ops)


def is_flaggems_available() -> bool:
    """Check if FlagGems is installed and available."""
    try:
        import flag_gems  # noqa: F401

        return True
    except ImportError:
        return False


def register_flagos_operator(
    op_name: str, impl_func: Callable, schema: Optional[str] = None
) -> None:
    """
    Register a FlagGems operator implementation for the flagos device.

    Args:
        op_name: The aten operator name (e.g., "add.Tensor")
        impl_func: The FlagGems implementation function
        schema: Optional schema string for custom operators
    """
    lib = torch.library.Library("aten", "IMPL")
    lib.impl(op_name, impl_func, "PrivateUse1")
    _registered_ops.add(op_name)
    if not hasattr(register_flagos_operator, "_libs"):
        register_flagos_operator._libs = []
    register_flagos_operator._libs.append(lib)


def enable_flaggems_for_flagos(
    exclude: Optional[List[str]] = None, include: Optional[List[str]] = None
) -> int:
    """
    Enable FlagGems operators for the flagos device.

    This function registers FlagGems' optimized Triton kernels as
    implementations for operations on flagos tensors.

    Note: FlagGems operators are automatically registered on import.
    Use this function only if you need to re-register with different options.

    Args:
        exclude: List of operator names to exclude
        include: List of operator names to include (if specified, only these are registered)

    Returns:
        Number of operators successfully registered
    """
    try:
        import flag_gems  # noqa: F401
    except ImportError:
        raise ImportError(
            "flag_gems is required for this function. "
            "Please install it with: pip install flag-gems"
        )

    # Create a library for PrivateUse1 dispatch key
    lib = torch.library.Library("aten", "IMPL")

    # Import the config to get all available operators
    from flag_gems import _FULL_CONFIG

    exclude = exclude or []
    registered_count = 0

    for item in _FULL_CONFIG:
        if len(item) < 2:
            continue

        op_name = item[0]
        impl_func = item[1]

        # Check include/exclude
        if include and op_name not in include:
            continue
        if op_name in exclude:
            continue

        # Check version conditions if present
        if len(item) > 2:
            condition = item[2]
            if callable(condition) and not condition():
                continue

        try:
            lib.impl(op_name, impl_func, "PrivateUse1")
            _registered_ops.add(op_name)
            registered_count += 1
        except Exception:
            # Some operators may already be registered or have incompatible signatures
            pass

    # Keep the library reference alive
    enable_flaggems_for_flagos._lib = lib
    return registered_count


class use_flaggems:
    """
    Context manager for temporarily enabling FlagGems operators on flagos device.

    Usage:
        with use_flaggems():
            x = torch.randn(100, 100, device="flagos")
            y = torch.matmul(x, x)  # Uses FlagGems matmul
    """

    def __init__(
        self, exclude: Optional[List[str]] = None, include: Optional[List[str]] = None
    ):
        self.exclude = exclude
        self.include = include
        self._lib = None

    def __enter__(self):
        self._lib = torch.library.Library("aten", "IMPL")

        try:
            import flag_gems  # noqa: F401
            from flag_gems import _FULL_CONFIG
        except ImportError:
            raise ImportError("flag_gems is required")

        exclude = self.exclude or []

        for item in _FULL_CONFIG:
            if len(item) < 2:
                continue

            op_name = item[0]
            impl_func = item[1]

            if self.include and op_name not in self.include:
                continue
            if op_name in exclude:
                continue

            if len(item) > 2:
                condition = item[2]
                if callable(condition) and not condition():
                    continue

            try:
                self._lib.impl(op_name, impl_func, "PrivateUse1")
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lib is not None:
            if hasattr(self._lib, "_destroy"):
                self._lib._destroy()
            del self._lib
        return False
