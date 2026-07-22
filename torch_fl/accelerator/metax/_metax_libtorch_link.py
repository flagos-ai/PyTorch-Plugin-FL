"""Symlink MetaX libtorch .so into the active (official) torch wheel's lib dir.

Rationale
---------
On MetaX we reuse PyTorch's CUDA boxing kernels (FLAGOS_METAX_BOXING) by running
the *MetaX* C++ runtime (libtorch_cpu.so / libtorch_cuda.so / libc10.so ...),
which is a hard fork exporting ``at::maca::*`` symbols.  When the Python front-end
is a *stock* ``torch==X.Y.Z+cpu`` wheel (no CUDA, clean pip env), its own
``torch/lib`` ships the upstream C++ .so.  We must make the process load the
MetaX C++ runtime instead.

Pure ``ctypes`` preloading does NOT reliably work: the official ``_C.so`` /
``libtorch_python.so`` carry an ``$ORIGIN`` RUNPATH that pulls the upstream
``libc10.so`` back in by full path, giving a *second* libc10 in the process and a
duplicate static-init crash (``Key already registered ... caffe2_report_cpu_memory_usage``).

The robust fix is to make the physical files the RUNPATH resolves to *be* the
MetaX ones -- i.e. replace the stock wheel's ``torch/lib/<so>`` with symlinks to
the MetaX wheel's copies.  Originals are backed up to ``torch/lib/_orig_backup/``
so the operation is fully reversible.

This runs from ``torch_fl/__init__.py`` BEFORE ``import torch`` (once torch is
imported its libc10 is already mapped and relinking is too late).  It is
idempotent, gated on ``FLAGOS_METAX_BOXING=1``, and a no-op when the active torch
already IS the MetaX wheel.
"""

import importlib.util
import os

# Core C++ .so that must come from the MetaX wheel as a self-consistent set.
# libtorch_python.so is included because the stock one references symbols
# (e.g. torch::jit::fuser::onednn::fuseGraph) absent from the MetaX libtorch_cpu.so.
_CORE_SO = (
    "libc10.so",
    "libtorch_cpu.so",
    "libtorch.so",
    "libtorch_global_deps.so",
    "libtorch_python.so",
)
# CUDA .so the stock +cpu wheel does not ship at all; symlinked in fresh.
_CUDA_SO = (
    "libc10_cuda.so",
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
)

_done = False


def _active_torch_lib():
    """torch/lib of the importable torch, WITHOUT importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    lib = os.path.join(spec.submodule_search_locations[0], "lib")
    return lib if os.path.isdir(lib) else None


def _bundled_maca_lib():
    """Forked libtorch bundled inside this wheel (self-contained MetaX build).

    ``scripts/bundle_maca_libtorch.sh`` copies the MetaX libtorch .so into
    ``torch_fl/lib_maca/``.  When present this is the preferred source: the
    target machine then needs only the official ``torch+cpu`` wheel plus the
    ``/opt/maca`` driver runtime, no separate MetaX torch wheel.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # this file: torch_fl/accelerator/metax/_metax_libtorch_link.py
    pkg_root = os.path.dirname(os.path.dirname(here))  # -> torch_fl/
    libdir = os.path.join(pkg_root, "lib_maca")
    if os.path.isdir(libdir) and os.path.exists(
        os.path.join(libdir, "libtorch_cuda.so")
    ):
        return libdir
    return None


def _discover_maca_torch_lib():
    """Locate the MetaX libtorch .so dir.

    Priority: forked libtorch bundled in this wheel (lib_maca/), then an
    explicit env var, then sibling conda envs whose torch is a
    ``+metax``/``+maca`` build (fallback for multi-env dev setups).
    """
    bundled = _bundled_maca_lib()
    if bundled:
        return bundled

    env = os.environ.get("FLAGOS_MACA_TORCH_LIB")
    if env and os.path.isdir(env):
        return env

    # Scan conda envs next to the current prefix for a MetaX torch build.
    prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(os.path.dirname(os.__file__))
    envs_root = os.path.dirname(prefix)  # .../envs
    if not os.path.isdir(envs_root):
        return None
    py = "python{}.{}".format(*__import__("sys").version_info[:2])
    for name in sorted(os.listdir(envs_root)):
        cand = os.path.join(envs_root, name, "lib", py, "site-packages", "torch")
        libdir = os.path.join(cand, "lib")
        ver_file = os.path.join(cand, "version.py")
        if not os.path.isfile(ver_file) or not os.path.isdir(libdir):
            continue
        try:
            with open(ver_file) as f:
                txt = f.read()
        except OSError:
            continue
        if ("metax" in txt or "maca" in txt) and os.path.exists(
            os.path.join(libdir, "libtorch_cuda.so")
        ):
            return libdir
    return None


def _link_one(dst_dir, backup_dir, name, target, required):
    """Idempotently point dst_dir/name at target (a MetaX .so)."""
    dst = os.path.join(dst_dir, name)
    if not os.path.exists(target):
        if required:
            raise FileNotFoundError(f"MetaX so missing: {target}")
        return
    # Already correctly linked?
    if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(target):
        return
    # Back up a real (non-symlink) original once.
    if os.path.exists(dst) and not os.path.islink(dst):
        os.makedirs(backup_dir, exist_ok=True)
        bak = os.path.join(backup_dir, name)
        if not os.path.exists(bak):
            os.replace(dst, bak)
        else:
            os.remove(dst)
    elif os.path.islink(dst):
        os.remove(dst)  # stale/incorrect link
    os.symlink(target, dst)


def ensure_maca_libtorch_links():
    """Symlink the active torch wheel's core .so to the MetaX wheel's copies.

    No-op unless FLAGOS_METAX_BOXING=1.  Idempotent; reversible via _orig_backup.
    Returns True if links are in place (or already were), False if skipped.
    """
    global _done
    if _done:
        return True
    if os.environ.get("FLAGOS_METAX_BOXING", "0") != "1":
        return False

    active = _active_torch_lib()
    maca = _discover_maca_torch_lib()
    if active is None or maca is None:
        return False
    # Already running on the MetaX wheel itself -> nothing to do.
    if os.path.realpath(active) == os.path.realpath(maca):
        _done = True
        return True

    backup = os.path.join(active, "_orig_backup")
    for name in _CORE_SO:
        _link_one(active, backup, name, os.path.join(maca, name), required=True)
    for name in _CUDA_SO:
        _link_one(active, backup, name, os.path.join(maca, name), required=False)
    _done = True
    return True


def restore_original_libtorch():
    """Undo ensure_maca_libtorch_links(): remove links, restore backups."""
    active = _active_torch_lib()
    if active is None:
        return
    backup = os.path.join(active, "_orig_backup")
    for name in _CORE_SO + _CUDA_SO:
        dst = os.path.join(active, name)
        if os.path.islink(dst):
            os.remove(dst)
    if os.path.isdir(backup):
        for name in os.listdir(backup):
            os.replace(os.path.join(backup, name), os.path.join(active, name))
        try:
            os.rmdir(backup)
        except OSError:
            pass
