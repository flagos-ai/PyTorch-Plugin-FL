import ctypes
import os


def _load_dll_libraries():
    """Load required DLL libraries on Windows."""
    lib_dir = os.path.join(os.path.dirname(__file__), "lib")
    if os.path.exists(lib_dir):
        for filename in os.listdir(lib_dir):
            if filename.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(lib_dir, filename))
                except OSError:
                    pass
