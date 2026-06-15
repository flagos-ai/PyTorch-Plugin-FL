#!/usr/bin/env python3
"""
Patch triton-ascend to work with torch_fl (without torch_npu dependency).

Original triton-ascend is designed to work with torch_npu. This script
patches it to use the torch_fl (flagos) device interface instead,
removing all hard dependencies on libtorch_npu.so.

Usage:
    python scripts/patch_triton_ascend.py [--triton-path /path/to/triton]

    If --triton-path is not given, auto-detects from `import triton`.
"""

import argparse
import os
import sys


def find_triton_path():
    """Auto-detect triton installation path."""
    try:
        import triton

        return os.path.dirname(triton.__file__)
    except ImportError:
        print(
            "ERROR: triton not found. Specify --triton-path explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)


def patch_file(filepath, replacements):
    """Apply (old, new) replacements to a file. Returns True if changed."""
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content
    applied = 0

    for old, new in replacements:
        if old not in content:
            if new in content:
                continue  # already patched
            print("  WARNING: pattern not found:")
            print(f"    {repr(old[:80])}")
            continue
        content = content.replace(old, new)
        applied += 1

    if content == original:
        print(f"  OK (already patched): {filepath}")
        return False

    with open(filepath, "w") as f:
        f.write(content)
    print(f"  PATCHED ({applied} changes): {filepath}")
    return True


def get_torch_npu_path(triton_path):
    """Determine torch_npu path for include headers."""
    site_packages = os.path.dirname(triton_path)
    return os.path.join(site_packages, "torch_npu")


def patch_driver(triton_path):
    """Patch backends/ascend/driver.py"""
    fp = os.path.join(triton_path, "backends", "ascend", "driver.py")

    replacements = [
        # --- Python API patches ---
        # get_current_device
        (
            "        import torch_npu\n        return torch.npu.current_device()",
            "        # import torch_npu  # patched for torch_fl\n"
            "        return torch.flagos.current_device()",
        ),
        # set_current_device
        (
            "        import torch_npu\n        return torch.npu.set_device(device)",
            "        # import torch_npu  # patched for torch_fl\n"
            "        return torch.flagos.set_device(device)",
        ),
        # get_current_stream: remove native _C import
        (
            "        import torch_npu\n"
            "        from torch_npu._C import"
            " _npu_getCurrentRawStream",
            "        # import torch_npu  # patched for torch_fl\n"
            "        # from torch_npu._C import"
            " _npu_getCurrentRawStream  # patched",
        ),
        (
            "        return _npu_getCurrentRawStream(device)",
            "        return 0  # default stream for torch_fl",
        ),
        # get_device_interface
        (
            "        return torch.npu\n",
            "        return torch.flagos\n",
        ),
        # get_empty_cache_for_benchmark
        (
            "device='npu')",
            "device='flagos')",
        ),
        # --- C++ template patches ---
        # Remove NPUWorkspaceAllocator.h
        (
            "#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>",
            "// torch_npu removed: workspace allocated via rtMalloc",
        ),
        # Replace at_npu allocator with rtMalloc
        (
            "    at::Tensor syncBlockLock_tensor = "
            "at_npu::native::allocate_workspace"
            "(syncBlockLockSize, stream);\n"
            "    syncBlockLock_ptr = const_cast<void *>"
            "(syncBlockLock_tensor.storage().data());",
            "    ret = rtMalloc(&syncBlockLock_ptr,"
            " syncBlockLockSize, RT_MEMORY_HBM);\n"
            "    ",
        ),
        # TRITON_ENABLE_TASKQUEUE default: true -> false
        # (taskqueue requires torch_npu OpCommand.h)
        (
            "\"TRITON_ENABLE_TASKQUEUE\", 'true'",
            "\"TRITON_ENABLE_TASKQUEUE\", 'false'",
        ),
    ]

    return patch_file(fp, replacements)


def patch_utils(triton_path):
    """Patch backends/ascend/utils.py"""
    fp = os.path.join(triton_path, "backends", "ascend", "utils.py")
    npu_path = get_torch_npu_path(triton_path)

    replacements = [
        # Remove -ltorch_npu linker flag
        (
            '        "-ltorch_npu",',
            '        # "-ltorch_npu",  # removed for torch_fl',
        ),
        # _npu_version_hash: hardcode version
        # (import torch_npu is not adjacent to torch_npu_version)
        (
            "    torch_npu_version = torch_npu.version.git_version",
            '    torch_npu_version = "torch_fl_shim"',
        ),
        # Replace dynamic torch_npu path with hardcoded
        # (import torch_npu is not adjacent to torch_npu_path)
        (
            "    torch_npu_path = os.path.dirname("
            "os.path.realpath(torch_npu.__file__))",
            '    torch_npu_path = "' + npu_path + '"',
        ),
    ]

    # Comment out all bare `import torch_npu` lines
    # (they appear after `import torch` in several functions)
    replacements.append(
        (
            "    import torch_npu\n",
            "    # import torch_npu  # patched for torch_fl\n",
        )
    )

    return patch_file(fp, replacements)


def main():
    parser = argparse.ArgumentParser(
        description="Patch triton-ascend for torch_fl compatibility"
    )
    parser.add_argument(
        "--triton-path",
        default=None,
        help="Path to triton package directory. Auto-detected if not specified.",
    )
    args = parser.parse_args()

    triton_path = args.triton_path or find_triton_path()
    print(f"Triton path: {triton_path}")

    if not os.path.isdir(os.path.join(triton_path, "backends", "ascend")):
        print(
            "ERROR: backends/ascend/ not found. Is this triton-ascend?",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\n[1/2] Patching backends/ascend/driver.py ...")
    patch_driver(triton_path)

    print("\n[2/2] Patching backends/ascend/utils.py ...")
    patch_utils(triton_path)

    print("\nDone. triton-ascend is now compatible with torch_fl.")
    print("NOTE: Clear triton kernel cache if you had previously compiled kernels:")
    print("  rm -rf ~/.triton/cache/")


if __name__ == "__main__":
    main()
