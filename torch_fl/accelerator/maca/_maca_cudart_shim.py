"""Load the MACA cudart shim before torch is imported.

On MetaX (MACA) hardware, PyTorch's .so files require CUDA runtime symbols
with @@libcudart.so.12 version tags.  MACA's libsymbol_cu.so provides these
symbols but without version tags.  We build and load a shim library
(csrc/runtime/accelerator/maca/csrc/cudart_shim.c) that re-exports all needed symbols with the correct
version tags.
"""

import ctypes
import os
import subprocess

_loaded = False


def ensure_cudart_shim():
    """Build (if needed) and load the cudart shim into the process."""
    global _loaded
    if _loaded:
        return

    # Detect MACA environment
    maca_path = os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME")
    if not maca_path:
        for candidate in ["/opt/maca", "/opt/maca-3.3.0"]:
            if os.path.isdir(candidate):
                maca_path = candidate
                break
    if not maca_path or not os.path.isdir(maca_path):
        return  # Not a MACA environment

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(pkg_dir))
    )  # project root
    csrc = os.path.join(base_dir, "csrc", "runtime", "accelerator", "maca", "csrc")
    build_dir = os.path.join(base_dir, "build")

    shim_src = os.path.join(csrc, "cudart_shim.c")
    version_script = os.path.join(csrc, "libcudart.version")

    if not os.path.exists(shim_src):
        # Source not available (installed wheel without csrc)
        return

    os.makedirs(build_dir, exist_ok=True)
    shim_so = os.path.join(build_dir, "libcudart_shim.so")

    # Rebuild if source is newer
    inputs = [shim_src, version_script]
    if not os.path.exists(shim_so) or any(
        os.path.exists(s) and os.path.getmtime(s) > os.path.getmtime(shim_so)
        for s in inputs
    ):
        subprocess.check_call(
            [
                "gcc",
                "-shared",
                "-fPIC",
                "-o",
                shim_so,
                shim_src,
                f"-Wl,--version-script={version_script}",
                "-Wl,-soname,libcudart.so.12",
                "-ldl",
            ]
        )

    ctypes.CDLL(shim_so, mode=ctypes.RTLD_GLOBAL)
    _loaded = True
