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

# Accelerator platform: "cuda" (default) or "maca"
ACCELERATOR = os.environ.get("ACCELERATOR", "cuda").lower()

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

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


def _maca_path_from_env() -> str:
    return (
        os.environ.get("MACA_PATH")
        or os.environ.get("MACA_HOME")
        or "/opt/maca"
    )


def _setup_maca_build_env(env: dict) -> str:
    """PATH/LD_LIBRARY_PATH for mxcc/cucc and MACA runtime. Returns MACA_PATH."""
    maca_path = _maca_path_from_env()
    cu_bridge = os.path.join(maca_path, "tools", "cu-bridge")
    cucc = os.path.join(cu_bridge, "bin", "cucc")
    if not os.path.isfile(cucc):
        raise RuntimeError(f"MACA cucc/mxcc not found: {cucc}")

    env.setdefault("MACA_PATH", maca_path)
    env["PATH"] = os.pathsep.join(
        p
        for p in (
            os.path.join(cu_bridge, "bin"),
            os.path.join(maca_path, "bin"),
            os.path.join(maca_path, "mxgpu_llvm", "bin"),
            env.get("PATH", ""),
        )
        if p
    )
    ld_parts = [
        os.path.join(maca_path, "lib"),
        os.path.join(cu_bridge, "lib"),
        os.path.join(maca_path, "mxgpu_llvm", "lib"),
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in ld_parts if p)
    return maca_path


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
    if ACCELERATOR == "maca":
        cmake_args.extend(
            [
                "-DMACA_KERNEL=ON",
                "-DCUDA_KERNEL=OFF",
                "-DFLAGGEMS_KERNEL=OFF",
            ]
        )

    # FlagGems C++ library path (optional, enables low-overhead C++ dispatch)
    flaggems_dir = os.environ.get("FLAGGEMS_DIR")
    if flaggems_dir:
        cmake_args.append(f"-DFlagGems_DIR={flaggems_dir}")

    build_env = os.environ.copy()
    build_jobs = _cmake_build_jobs()
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_jobs)
    cmake = "cmake"

    if ACCELERATOR == "maca":
        maca_path = _setup_maca_build_env(build_env)
        cmake_args.append(f"-DMACA_PATH={maca_path}")
        cmake_args.append("-G")
        cmake_args.append("Ninja")
    else:
        # Add CUDA toolkit path if available
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc")

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
    if ACCELERATOR != "maca":
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
            "Remove build/ and torch_fl/lib/*.so, then rebuild with ACCELERATOR=maca."
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
