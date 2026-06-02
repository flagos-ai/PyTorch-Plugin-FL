import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

# Accelerator platform: "cuda" (default), "metax", or "ascend"
ACCELERATOR = os.environ.get("ACCELERATOR", "cuda").lower()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Only run cmake build for actual build commands, not metadata collection
BUILD_COMMANDS = {
    "build",
    "build_ext",
    "install",
    "develop",
    "bdist_wheel",
    "bdist_egg",
    "editable_wheel",
}
RUN_BUILD_DEPS = any(arg in BUILD_COMMANDS for arg in sys.argv)


def _ensure_metax_cudart_shim():
    """On MetaX, compile and load a complete cudart shim before importing torch.

    MetaX's libsymbol_cu.so provides CUDA runtime symbols but without the
    @@libcudart.so.12 version tags that PyTorch's .so files require.
    We build a single shared library (csrc/runtime/accelerator/metax/cudart_shim.c) that:
      1. Forwards ~79 symbols to libsymbol_cu.so via dlsym
      2. Stubs ~11 symbols for APIs missing from MetaX entirely
      3. Tags ALL exported symbols with @@libcudart.so.12 via a version script
    """
    import ctypes

    csrc = os.path.join(BASE_DIR, "csrc", "runtime", "accelerator", "metax")
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    shim_so = os.path.join(build_dir, "libcudart_shim.so")
    shim_src = os.path.join(csrc, "cudart_shim.c")
    version_script = os.path.join(csrc, "libcudart.version")

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


if ACCELERATOR == "metax":
    _ensure_metax_cudart_shim()


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def _cuda_toolkit_root() -> str | None:
    """Locate CUDA toolkit root (directory containing include/cuda_runtime.h)."""
    candidates: list[str] = []
    for key in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(key)
        if val:
            candidates.append(val)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "targets", "x86_64-linux"),
                conda_prefix,
            ]
        )
    candidates.append("/usr/local/cuda")

    seen: set[str] = set()
    for root in candidates:
        root = os.path.realpath(root)
        if root in seen:
            continue
        seen.add(root)
        if os.path.isfile(os.path.join(root, "include", "cuda_runtime.h")):
            return root
    return None


def _find_nvcc(cuda_root: str) -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    for candidate in (
        os.path.join(cuda_root, "bin", "nvcc"),
        os.path.join(conda_prefix, "bin", "nvcc") if conda_prefix else None,
        shutil.which("nvcc"),
    ):
        if candidate and os.path.isfile(candidate):
            return os.path.realpath(candidate)
    return None


def _prepend_env_path(env: dict, key: str, *paths: str) -> None:
    parts = [p for p in paths if p and os.path.isdir(p)]
    existing = env.get(key, "")
    if existing:
        parts.append(existing)
    if parts:
        env[key] = os.pathsep.join(parts)


def _pip_nvidia_include_dirs() -> list[str]:
    """Headers from pip nvidia-* wheels when conda toolkit is minimal."""
    import pathlib
    import site

    dirs: list[str] = []
    for sp in site.getsitepackages():
        nvidia = pathlib.Path(sp) / "nvidia"
        if not nvidia.is_dir():
            continue
        for pkg in sorted(nvidia.iterdir()):
            inc = pkg / "include"
            if inc.is_dir():
                dirs.append(str(inc))
    return dirs


def _setup_cuda_build_env(env: dict) -> str | None:
    """Export CUDA paths for cmake/nvcc (incl. conda pip wheel layout)."""
    cuda_root = _cuda_toolkit_root()
    if not cuda_root:
        return None

    env.setdefault("CUDA_HOME", cuda_root)
    env.setdefault("CUDA_PATH", cuda_root)
    _prepend_env_path(env, "CPATH", os.path.join(cuda_root, "include"))
    _prepend_env_path(env, "CPATH", *_pip_nvidia_include_dirs())
    _prepend_env_path(env, "LIBRARY_PATH", os.path.join(cuda_root, "lib"))
    _prepend_env_path(env, "LD_LIBRARY_PATH", os.path.join(cuda_root, "lib"))
    _prepend_env_path(env, "CMAKE_PREFIX_PATH", cuda_root)
    return cuda_root


def _find_nvrtc_library() -> str | None:
    try:
        import importlib.util
        import pathlib

        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec is None or not spec.origin:
            return None
        lib = pathlib.Path(spec.origin).resolve().parent / "lib" / "libnvrtc.so.12"
        return str(lib) if lib.is_file() else None
    except Exception:
        return None


def _append_cuda_cmake_args(cmake_args: list[str], cuda_root: str) -> None:
    nvcc = _find_nvcc(cuda_root)
    if nvcc:
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
    cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")
    cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}")
    nvrtc = _find_nvrtc_library()
    if nvrtc:
        cmake_args.append(f"-DCUDA_nvrtc_LIBRARY={nvrtc}")


def _find_flaggems_dir() -> str | None:
    env_dir = os.environ.get("FLAGGEMS_DIR")
    if env_dir and os.path.isfile(os.path.join(env_dir, "FlagGemsConfig.cmake")):
        return env_dir

    import site

    search_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        search_roots.append(user_site)
    for sp in search_roots:
        cand = os.path.join(sp, "flag_gems", "lib", "cmake", "FlagGems")
        if os.path.isfile(os.path.join(cand, "FlagGemsConfig.cmake")):
            return cand
    return None


def _metax_path_from_env() -> str:
    return (
        os.environ.get("METAX_PATH")
        or os.environ.get("METAX_HOME")
        or os.environ.get("MACA_PATH")
        or os.environ.get("MACA_HOME")
        or "/opt/maca"
    )


def _setup_metax_build_env(env: dict) -> str:
    """PATH/LD_LIBRARY_PATH for mxcc/cucc and MetaX runtime. Returns METAX_PATH."""
    metax_path = _metax_path_from_env()
    cu_bridge = os.path.join(metax_path, "tools", "cu-bridge")
    cucc = os.path.join(cu_bridge, "bin", "cucc")
    if not os.path.isfile(cucc):
        raise RuntimeError(f"MetaX cucc/mxcc not found: {cucc}")

    env.setdefault("METAX_PATH", metax_path)
    env["PATH"] = os.pathsep.join(
        p
        for p in (
            os.path.join(cu_bridge, "bin"),
            os.path.join(metax_path, "bin"),
            os.path.join(metax_path, "mxgpu_llvm", "bin"),
            env.get("PATH", ""),
        )
        if p
    )
    ld_parts = [
        os.path.join(metax_path, "lib"),
        os.path.join(cu_bridge, "lib"),
        os.path.join(metax_path, "mxgpu_llvm", "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in ld_parts if p)
    return metax_path


def _cmake_build_jobs() -> int:
    """Parallel compile jobs for cmake/ninja. Set FLAGOS_BUILD_JOBS=1 for serial logs."""
    for key in ("FLAGOS_BUILD_JOBS", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            jobs = int(raw)
            if jobs < 1:
                raise ValueError(f"{key} must be >= 1, got {raw!r}")
            return jobs
    return multiprocessing.cpu_count()


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_fl")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    cmake_args.append(f"-DACCELERATOR={ACCELERATOR}")
    if ACCELERATOR == "metax":
        cmake_args.extend(
            [
                "-DMETAX_KERNEL=ON",
                "-DCUDA_KERNEL=OFF",
                "-DFLAGGEMS_KERNEL=OFF",
            ]
        )

    # Kernel build options from environment
    for kernel_opt in (
        "FLAGGEMS_KERNEL",
        "FLAGGEMS_PYTHON",
        "CUDA_KERNEL",
        "METAX_KERNEL",
        "ASCEND_KERNEL",
    ):
        val = os.environ.get(kernel_opt)
        if val is not None:
            cmake_val = (
                "ON" if val not in ("0", "OFF", "off", "false", "FALSE") else "OFF"
            )
            cmake_args.append(f"-D{kernel_opt}={cmake_val}")

    build_env = os.environ.copy()
    build_jobs = _cmake_build_jobs()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_jobs)
    cmake = "cmake"

    # FlagGems C++ library path (optional, enables low-overhead C++ dispatch)
    flaggems_dir = os.environ.get("FLAGGEMS_DIR")
    if flaggems_dir:
        cmake_args.append(f"-DFlagGems_DIR={flaggems_dir}")
    flaggems_source_dir = os.environ.get("FLAGGEMS_SOURCE_DIR")
    if flaggems_source_dir:
        cmake_args.append(f"-DFLAGGEMS_SOURCE_DIR={flaggems_source_dir}")

    if ACCELERATOR == "metax":
        metax_path = _setup_metax_build_env(build_env)
        cmake_args.append(f"-DMETAX_PATH={metax_path}")
        cmake_args.append("-G")
        cmake_args.append("Ninja")
    elif ACCELERATOR == "cuda":
        cuda_root = _setup_cuda_build_env(build_env)
        if cuda_root:
            _append_cuda_cmake_args(cmake_args, cuda_root)
        flaggems_dir = _find_flaggems_dir()
        if flaggems_dir:
            cmake_args.append(f"-DFLAGGEMS_DIR={flaggems_dir}")

    subprocess.check_call([cmake, BASE_DIR] + cmake_args, cwd=build_dir, env=build_env)

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(build_jobs)]
    else:
        build_args += ["-j", str(build_jobs)]

    subprocess.check_call([cmake] + build_args, cwd=build_dir, env=build_env)
    _verify_built_native_libs()


def _verify_built_native_libs() -> None:
    lib = os.path.join(BASE_DIR, "torch_fl", "lib", "libtorch_fl.so")
    if not os.path.isfile(lib):
        raise RuntimeError(
            f"Native build finished but {lib} is missing. "
            "Check cmake/ninja output above."
        )
    if ACCELERATOR != "metax":
        return
    try:
        undef = subprocess.check_output(
            ["nm", "-u", lib], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return
    if "get_maca_enable_elementwise_kernel_info" in undef:
        raise RuntimeError(
            f"{lib} still references at::maca::* (mcPytorch). "
            "Remove build/ and torch_fl/lib/*.so, then rebuild with ACCELERATOR=metax."
        )


class BuildExtWithCmake(_build_ext):
    """Run cmake before setuptools builds torch_fl._C."""

    def run(self):
        build_deps()
        super().run()


class EditableWheelWithCmake(_editable_wheel):
    """PEP 660 editable installs must build native libs (pip often skips build_ext)."""

    def run(self):
        self.run_command("build_ext")
        super().run()


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_fl/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_fl")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def _extension_compile_args():
    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]
    return extra_link_args, extra_compile_args


def _get_setup_kwargs():
    extra_link_args, extra_compile_args = _extension_compile_args()
    ext_modules = [
        Extension(
            name="torch_fl._C",
            sources=["torch_fl/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_fl/lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_fl": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
            "backends.conf",
        ]
    }

    return dict(
        name="torch_fl",
        version="0.1.0",
        description="FlagGems operators as a custom PyTorch device (flagos)",
        author="FlagGems Team",
        packages=find_packages(
            include=["torch_fl*", "accelerator*", "csrc.runtime.accelerator*"]
        ),
        package_dir={"": "."},
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "build_ext": BuildExtWithCmake,
            "editable_wheel": EditableWheelWithCmake,
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        python_requires=">=3.8",
        install_requires=[
            "torch",
        ],
    )


# PEP 517 / pip install -e loads setup.py as a script; setup() must run at import time
# so cmdclass (build_ext / editable_wheel) is registered. Do not hide setup() in main().
setup(**_get_setup_kwargs())
