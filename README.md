# torch_fl

A custom PyTorch device plugin based on the PrivateUse1 extension mechanism, registering [FlagGems](https://github.com/FlagOpen/FlagGems) high-performance Triton operators as the `flagos` device backend for unified multi-chip support.

## Features

- Automatically registers FlagGems Triton operators as dispatch implementations for the `flagos` device
- Configurable backend routing: select FlagGems or native vendor backend (CUDA/MetaX/Ascend) at per-operator granularity
- Currently supports CUDA, MetaX, and Ascend hardware platforms
- Complete device management API (stream, event, RNG, AMP)
## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.12 |
| PyTorch | 2.11.0 |
| CUDA | 12.8 |
| FlagGems | 5.0.2 |

> CUDA 12.2 has known numerical precision issues (NaN). Please use version 12.9 or higher.

## Installation

### Prerequisites

- Hardware Runtime Dependencies:
    - CUDA toolkit 12.8 (required only on CUDA platform)
    - MetaX cu-bridge library (required only on MetaX platform; from the [MetaX developer portal](https://developer.metax-tech.com/softnova))
    - CANN toolkit (required only on Ascend platform)
- PyTorch 2.11.0
- FlagGems (version 5.0.2 or higher, requires DFLAGGEMS_BUILD_C_EXTENSIONS enabled). For source installation, refer to: [FlagGems Installation](https://flagos-ai.github.io/FlagGems/getting-started/install/)
  - Note: FlagGems is optional on Ascend platform

### Build from Source (CUDA Platform)

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

ACCELERATOR=cuda FLAGGEMS_DIR=/path/to/FlagGems/build/cpython-312/ \
  FLAGGEMS_KERNEL=1 FLAGGEMS_PYTHON=1 CUDA_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

### Build from Source (MetaX Platform)

MetaX ships a **self-contained boxing wheel**: it reuses PyTorch's generated CUDA boxing kernels (host `g++`, no `mxcc`) and bundles the forked libtorch C++ runtime inside the wheel. The target machine then needs only:

- The official `torch==2.10.0+cpu` wheel (from PyPI, no CUDA)
- This `torch_fl` wheel
- The `/opt/maca` driver runtime (present on any machine with a MetaX card)

No separate `torch+metax` wheel and no manual `LD_LIBRARY_PATH` are required — `import torch_fl` symlinks the stock wheel's `torch/lib` to the bundled forked libtorch, whose RPATH resolves the MetaX runtime under `/opt/maca`.

> **Getting the MetaX MACA SDK and `torch+metax` wheel** (needed only to *build* the wheel, not to run it)
> Both are distributed through the MetaX developer portal (SoftNova): <https://developer.metax-tech.com/softnova>. Registration/login is required. Download the MACA SDK (driver + cu-bridge) matching your card and driver version, and the `torch+metax` (`maca-pytorch`) wheel built for the same MACA version and your Python version — it is the source of the forked libtorch bundled into the wheel. Install the SDK to `/opt/maca` (or point `METAX_PATH` at the install location).

**Build the wheel** (on a machine with the MetaX SDK and a `torch+metax` wheel available as the libtorch source):

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# 1. Build the boxing artifacts (METAX_KERNEL is forced OFF in boxing mode)
ACCELERATOR=metax FLAGOS_METAX_BOXING=1 \
  FLAGOS_MACA_TORCH_LIB=/path/to/torch+metax/torch/lib \
  FLAGOS_WHEEL_LOCAL=metax3.8.1 \
  python setup.py bdist_wheel

# 2. Bundle the 8 forked libtorch .so into torch_fl/lib_maca/ and rewrite RPATH
#    (requires patchelf: pip install patchelf)
FLAGOS_MACA_TORCH_LIB=/path/to/torch+metax/torch/lib \
  MACA_PATH=/opt/maca \
  bash scripts/bundle_maca_libtorch.sh

# 3. Repackage so the bundled libtorch is included (reuses the built artifacts)
python setup.py build_py
cp build/lib.*/torch_fl/_C.*.so build/lib.*/torch_fl/  # ensure the C ext is staged
python setup.py bdist_wheel --skip-build --bdist-dir "$(mktemp -d)"
```

The result is `dist/torch_fl-0.1.0+metax3.8.1-cp312-cp312-linux_x86_64.whl` (~1.1 GB — it bundles the forked libtorch, so it exceeds the PyPI 100 MB limit and must be distributed via a private index or directly).

`FLAGOS_WHEEL_LOCAL` sets the local version segment (e.g. `metax3.8.1` → `0.1.0+metax3.8.1`) to tag the wheel with the target MACA/driver version.

**Install and run** on the target machine (clean env, MetaX card present):

```bash
pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_fl-0.1.0+metax3.8.1-cp312-cp312-linux_x86_64.whl

export FLAGOS_METAX_BOXING=1
python -c "
import torch_fl, torch          # torch_fl must be imported first
x = torch.randn(4, 4, device='flagos:0')
print((x + x).sum().cpu())
"
```

In boxing mode, `import torch_fl` auto-selects `backends_cuda.conf` (override with `FLAGOS_BACKEND_CONFIG`). If MACA is installed somewhere other than `/opt/maca`, pass `MACA_PATH` at bundle time (step 2) so the RPATH points there.

**Optional: FlagGems on MetaX.** Like the CUDA wheel, the MetaX boxing wheel compiles the FlagGems Python-path kernels (`flagos_python` backend) by default, so FlagGems is a runtime switch — set `FLAGOS_USE_FLAGGEMS=1` to route ops to FlagGems' Triton kernels where available, or leave it unset for pure CUDA-kernel reuse (boxing). Enabling it needs two extra target-side pip installs (not bundled in the wheel):

```bash
# On the target MetaX machine, in addition to the wheel + torch+cpu above:
pip install triton-metax flag_gems     # triton-metax emits mcfatbin for MetaX GPUs

export FLAGOS_METAX_BOXING=1
export FLAGOS_USE_FLAGGEMS=1            # opt into FlagGems; unset = pure boxing
python -c "import torch_fl, torch; x=torch.randn(1024, device='flagos:0'); print(torch.nn.functional.silu(x).sum().cpu())"
```

`import torch_fl` then auto-selects `backends_metax_flaggems.conf` and sets `GEMS_VENDOR=metax` + the MetaX `torch.cuda` compat shim automatically. That conf mirrors the CUDA `backends_flaggems.conf` but routes the ops triton-metax cannot run (`mm`/`bmm`/`mean.dim` — FlagGems uses a SPLIT_K kwarg / CUDA-context path triton-metax rejects) back to the `cuda` boxing kernel (maca `libtorch_cuda`), not the mxcc backend (which is off in boxing mode). Without `triton-metax`/`flag_gems` installed, leave `FLAGOS_USE_FLAGGEMS` unset — the pure boxing path has no extra dependencies.

### Build from Source (Ascend Platform)

#### 1. Install FlagGems (FLAGOS Backend)

On Ascend, FlagGems must be installed from our fork (`torch_fl` branch) with `FLAGGEMS_BACKEND=FLAGOS`. This avoids the `libtorch_npu.so` dependency — instead, FlagGems obtains the ACL stream via `torch_fl`'s `GetCurrentStream` C API.

```bash
# Clone FlagGems (torch_fl branch)
git clone -b torch_fl https://github.com/Hchnr/FlagGems.git
cd FlagGems

# Ensure CANN toolkit environment is sourced
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install FlagGems with FLAGOS backend (skip C++ extensions)
pip install --no-build-isolation -e . \
  --config-settings=cmake.define.FLAGGEMS_BACKEND=FLAGOS \
  --config-settings=cmake.define.FLAGGEMS_BUILD_C_EXTENSIONS=OFF

cd ..
```

> **Why FLAGOS backend?**
> The default ascend/npu backend links against `libtorch_npu.so`, which doesn't exist in our environment (`torch_fl` is the PrivateUse1 backend, not `torch_npu`).
> The `FLAGOS` backend resolves device streams via `extern "C" void* GetCurrentStream(int)`, provided by `torch_fl`'s `libstream_api.so`.

#### 2. Install torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

source /usr/local/Ascend/ascend-toolkit/set_env.sh

ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

Notes:
- `FLAGGEMS_KERNEL=0`: Disable FlagGems C++ kernel wrappers (FLAGOS backend does not compile `liboperators.so`)
- `FLAGGEMS_PYTHON=1`: Enable FlagGems Python wrappers to route ops to FlagGems Triton kernels
- `ASCEND_KERNEL=1`: Compile the Ascend C++ operator backend (ACL NN API)

#### 3. Patch triton-ascend

The stock triton-ascend package depends on `torch_npu` / `libtorch_npu.so`. Since `torch_fl` replaces `torch_npu` as the PrivateUse1 backend, we need to patch triton-ascend to use the `flagos` device interface instead:

```bash
python scripts/patch_triton_ascend.py
```

The script auto-detects the triton install path and applies the necessary changes. It is idempotent — running it multiple times is safe. After patching, clear any stale kernel cache:

```bash
rm -rf ~/.triton/cache/
```

#### 4. Verify Installation

```bash
python -c "
import torch_fl
print('device count:', torch_fl.flagos.device_count())
print('FlagGems enabled:', torch_fl.is_flaggems_enabled())
print('registered ops:', len(torch_fl.get_registered_ops()))
"
```

#### 5. Run Tests

```bash
# Inference test
pytest tests/integration/test_qwen3_infer.py -v -s --model /path/to/Qwen3-0.6B

# Training test
pytest tests/integration/test_qwen3_train.py -v -s --model /path/to/Qwen3-0.6B
```

> **Troubleshooting: `libtorch_npu.so: cannot open shared object file`**
>
> This error means triton-ascend is still trying to load `torch_npu`. Verify that:
> 1. You ran `python scripts/patch_triton_ascend.py` after installing triton-ascend
> 2. FlagGems was installed from `https://github.com/Hchnr/FlagGems` branch `torch_fl`
> 3. It was built with `FLAGGEMS_BACKEND=FLAGOS`
> 4. The triton kernel cache was cleared (`rm -rf ~/.triton/cache/`)

### Build from Source (PPU Platform)

PPU (`PPU_SDK`) presents itself as a **CUDA-compatible** device, so it reuses the
CUDA build directly — no dedicated `ACCELERATOR=ppu` branch is needed. The PPU
`torch` wheel is already a full CUDA-enabled build (`torch.version.cuda == '13.0'`,
`torch.cuda.is_available() == True`), and `PPU_SDK/CUDA_SDK` is a complete CUDA 13
toolkit (nvcc, headers, `libcudart.so.13`). This makes PPU the simplest case of the
boxing route:

- **No stock `+cpu` wheel and no external `libtorch_cuda.so`** are required — the
  PPU torch wheel ships its own CUDA runtime, loaded normally by `import torch`.
- PPU registers ops under the independent `CUDA` dispatch key (not `PrivateUse1`),
  so the generated CUDA boxing kernels (`csrc/aten/generated/cuda_kernels.cc`,
  PrivateUse1 → CUDA) are reused unchanged.

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# Pure CUDA-boxing build (no FlagGems). CUDA_HOME points at the PPU CUDA_SDK;
# FLAGOS_SKIP_CUDA_ASSETS=1 skips bundling an external libtorch_cuda.so AND skips
# the pinned nvidia-*-cu12 runtime deps (PPU supplies CUDA 13 via PPU_SDK/CUDA_SDK).
ACCELERATOR=cuda \
  CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK \
  CUDA_KERNEL=ON FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF \
  FLAGOS_SKIP_CUDA_ASSETS=1 \
  pip install --no-build-isolation -vvv -e .
```

At runtime, set `FLAGOS_DISABLE_CUDA_ASSETS=1` so the import-time preload of a
bundled `libtorch_cuda.so` is a no-op (there is none — PPU torch provides it):

```bash
FLAGOS_DISABLE_CUDA_ASSETS=1 python -c "import torch_fl, torch; \
  x = torch.randn(4, 4, device='flagos'); \
  print((x @ x).cpu())"
```

**Optional: FlagGems on PPU.** Set `FLAGGEMS_PYTHON=ON` at build time (the
default) and `FLAGOS_USE_FLAGGEMS=1` at runtime; `import torch_fl` then selects
`backends_flaggems.conf` and routes the discovered ops to FlagGems' Triton
kernels. PPU needs no compat shim beyond the generic CUDA one: `libcuda.so` is a
real driver, so `is_nvidia_cuda_available()` succeeds, `GEMS_VENDOR=nvidia` is
set, and `triton.language.extra.cuda.libdevice` resolves (it has `pow`).

PPU's Triton comes from the vendor index, not PyPI, and its version string
(`3.5.0+v0.2.0.ppu2.1.0`) does not satisfy the `triton>=3.5.1` pin — so when
`PPU_SDK` is set, `setup.py` drops the `flag_gems`/`triton` requirements and you
install them yourself:

```bash
pip install triton==3.5.0+v0.2.0.ppu2.1.0   # vendor index; see note below
pip install flag_gems

FLAGOS_DISABLE_CUDA_ASSETS=1 FLAGOS_USE_FLAGGEMS=1 python -c "import torch_fl, torch; \
  x = torch.randn(256, 256, device='flagos'); \
  print(torch.allclose(torch.softmax(x, -1).cpu(), torch.softmax(x.cpu(), -1), atol=1e-3))"
```

> **Troubleshooting: `Invalid cross-device link` installing PPU Triton**
>
> The vendor `triton` sdist is a downloader shim that fetches the real wheel and
> `rename()`s it into pip's cache. If your pip cache and build dir are on
> different filesystems (e.g. cache on NFS, build in `/tmp`) that rename fails
> with `[Errno 18]`. Fetch the wheel the shim reports (`Guessing wheel URL: ...`)
> with `curl` and `pip install` the file directly.

**Run tests:**

```bash
# Pure boxing
FLAGOS_DISABLE_CUDA_ASSETS=1 pytest tests/unit tests/integration/ops \
  tests/integration/test_factory_ops.py -q -m "not flaggems and not flaggems_python"

# FlagGems path (first run is slow: Triton compiles/autotunes every kernel)
FLAGOS_DISABLE_CUDA_ASSETS=1 FLAGOS_USE_FLAGGEMS=1 pytest tests/integration/ops -q
```

### Build from Source (Hygon DCU Platform)

Hygon DCU (DTK) reuses the **CUDA boxing route** with a dedicated
`ACCELERATOR=dcu` branch. Two properties of the vendor stack make this work:

- The DCU `torch` wheel is a **hipified** build: it registers its HIP kernels
  under the `CUDA` dispatch key and its tensors report `DeviceType::CUDA`
  (`torch.version.cuda is None`, `torch.version.hip == '6.3.x'`). So the
  generated PrivateUse1 → CUDA boxing kernels
  (`csrc/aten/generated/cuda_kernels.cc`) dispatch into `libtorch_hip.so`
  unchanged.
- DTK ships a **CUDA compatibility toolkit** at `$DTK_ROOT/cuda/cuda-*` whose
  `libcudart.so.12` is a thin shim over `libgalaxyhip.so` — the same runtime
  `libtorch_hip.so` uses, so there is one driver state, not two. The runtime
  sources under `csrc/runtime/accelerator/cuda/` therefore compile as-is with
  plain host `g++`; no `nvcc`, no `hipcc`, no hipify pass.

The build is **pure boxing**: `CUDA_KERNEL`, `FLAGGEMS_KERNEL` and
`FLAGGEMS_PYTHON` are all forced off (DTK ships its own Triton, so the
NVIDIA-targeted PyPI `triton` wheel is the wrong artifact and is not pulled in).

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

source /opt/dtk/env.sh          # exports ROCM_PATH; DTK_ROOT also honored

ACCELERATOR=dcu pip install --no-build-isolation -vvv -e .
```

`DTK_ROOT` resolves from `DTK_ROOT` → `ROCM_PATH` → `/opt/dtk`. Pass it
explicitly if DTK lives elsewhere.

**Verify:**

```bash
python -c "
import torch, torch_fl
print('device count:', torch.flagos.device_count())
x = torch.randn(512, 512, device='flagos')
print('mm matches .cuda():',
      torch.allclose(torch.mm(x, x).cpu(), torch.mm(x.cpu().cuda(), x.cpu().cuda()).cpu()))
"
```

**Run tests** (deselect the FlagGems markers — this is a pure-boxing build):

```bash
pytest tests/unit tests/integration/test_allocator.py tests/integration/test_factory_ops.py -q
pytest tests/integration/ops -q -m "not flaggems and not flaggems_python"
```

Notes:

- **Memory pool.** flagos tensors and the boxed kernels' outputs share one pool.
  `dcu_memory.h` delegates caching to torch's own allocator through the
  device-generic registry (`c10::getDeviceAllocator(kCUDA)`) rather than the
  `c10::cuda::` namespace — the DCU wheel exports `c10::hip::HIPCachingAllocator`
  and has zero `c10::cuda` symbols, and `cuda_runtime.h` cannot share a
  translation unit with `hip/hip_runtime.h`. So `memory_allocated()` /
  `memory_reserved()` / `empty_cache()` report and act on real usage.
- **`record_stream` is a no-op** on DCU: building a `c10::Stream` from a raw
  stream handle needs `c10::cuda::getStreamFromExternal`, which this wheel does
  not export.
- **`.cuda()` autograd and `torch_fl` cannot share a process.** PyTorch's
  `register_privateuse1_backend` makes `at::getAccelerator()` return
  `PrivateUse1`, so the autograd engine finds no stream metadata for a
  pure-CUDA graph and asserts in `engine.cpp`. This is upstream PrivateUse1
  behaviour, identical on CUDA/MetaX/Ascend — flagos-device autograd is
  unaffected. Take `.cuda()` baselines in a separate process.
- **DTK's exported MIOpen CMake config** bakes in `/usr/lib/x86_64-linux-gnu/librt.so`,
  which no longer exists on glibc ≥ 2.34 (librt was folded into libc). The
  `ACCELERATOR=dcu` branch rewrites that dangling absolute path to `-lrt`.

### Build Environment Variables

| Variable | Description |
|----------|-------------|
| `ACCELERATOR` | Hardware platform: `cuda` (default), `metax`, `ascend`, `tsingmicro`, or `dcu` |
| `CUDA_HOME` | CUDA toolkit path |
| `DTK_ROOT` | Hygon DTK path (falls back to `ROCM_PATH`, then `/opt/dtk`; required for DCU build) |
| `METAX_PATH` | MetaX SDK path (default `/opt/maca`; required for MetaX build) |
| `METAX_ARCH` / `METAX_MXCC` | Optional GPU arch or `mxcc`/`cucc` compiler path |
| `METAX_KERNEL` | Enable MetaX C++ kernel build (`ON`/`OFF`; auto-enabled when `ACCELERATOR=metax`) |
| `ASCEND_HOME` | CANN toolkit path (default `/usr/local/Ascend/ascend-toolkit/latest`) |
| `FLAGGEMS_DIR` | FlagGems C++ library path (enables low-overhead C++ dispatch) |
| `FLAGGEMS_KERNEL` | Enable FlagGems C++ kernel wrappers (`ON`/`OFF`, default `ON`; set `0` for Ascend) |
| `FLAGGEMS_PYTHON` | Enable FlagGems Python kernel wrappers (`ON`/`OFF`, default `OFF`; set `1` to enable) |
| `CUDA_KERNEL` | Enable CUDA kernel build (`ON`/`OFF`, default `ON`; set `0` for Ascend) |
| `ASCEND_KERNEL` | Enable Ascend kernel build (`ON`/`OFF`, default `OFF`; set `1` for Ascend) |

### Runtime Environment Variables

| Variable | Description |
|----------|-------------|
| `FLAGOS_METAX_CUDART_SHIM` | Set to `1` to preload libcudart compatibility shim before `import torch` (often needed with generic PyTorch wheels) |
| `FLAGOS_METAX_COMPAT` | Set to `1` to patch FlagGems `torch.cuda` device queries for MetaX |
| `GEMS_VENDOR` | FlagGems vendor name; set to `metax` on MetaX |
| `LD_PRELOAD` | Often set to `/opt/maca/lib/libsymbol_cu.so` for cu-bridge symbol resolution |
| `FLAGGEMS_SOURCE_DIR` | FlagGems source directory (required when ops route to `flaggems` or `flagos_python`) |
| `FLAGOS_BACKEND_CONFIG` | Override backend routing config (MetaX: `backends_metax.conf` or `backends_metax_flagos_py.conf`) |
| `FLAGOS_DISABLE_FLAGGEMS_PY` | Set to `1` to disable FlagGems Python-layer registration (C++ stub-only mode) |
| `FLAGOS_LOG_DISPATCH` | Set to `1` to print backend selection for each operator dispatch |
| `FLAGOS_OP_<name>` | Per-operator backend override (replace `.` with `__` in op names) |

## Usage

### Basic Usage

```python
import torch
import torch_fl  # Import automatically registers FlagGems operators

# Create tensors on flagos device
x = torch.randn(1000, 1000, device="flagos")
y = torch.randn(1000, 1000, device="flagos")

# All operations automatically use FlagGems Triton kernels
z = x + y
mm_result = torch.mm(x, y)
softmax_result = torch.softmax(x, dim=-1)
```

### Data Transfer Between Devices

```python
cpu_tensor = torch.randn(3, 3)
flagos_tensor = cpu_tensor.to("flagos")
back_to_cpu = flagos_tensor.cpu()
```

### Device Context Management

```python
with torch_fl.flagos.device(0):
    a = torch.randn(10, 10, device="flagos")
```

### MetaX Platform Import Order

On MetaX hardware, you **must** import `torch_fl` before `import torch`:

```python
import torch_fl  # Must import first
import torch
```

Reason: PyTorch's bundled CUDA 12.x runtime is ABI-incompatible with MetaX's cu-bridge (CUDA 11.6 compatibility layer). `torch_fl` preloads a shim library to provide the required symbol versions.

This restriction does not apply to CUDA platforms.

### MetaX Runtime Setup

Before running tests or inference on MetaX, source the SDK paths and hybrid backend config:

```bash
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so

export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_METAX_COMPAT=1
export GEMS_VENDOR=metax
export FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax_flagos_py.conf
export FLAGGEMS_SOURCE_DIR=$(python -c "import os,flag_gems;print(os.path.dirname(flag_gems.__file__))")
```

#### MetaX runtime notes

- **Boxing wheel**: The [self-contained boxing wheel](#build-from-source-metax-platform) (`FLAGOS_METAX_BOXING=1`) reuses PyTorch's CUDA boxing kernels and bundles the forked libtorch, running on official `torch+cpu` with no `mxcc`. By default ops route to `cuda` via `backends_cuda.conf` (no Triton, no extra deps). Setting `FLAGOS_USE_FLAGGEMS=1` opts into the FlagGems `flagos_python` path (`backends_metax_flaggems.conf`), which requires target-side `triton-metax` + `flag_gems`; ops triton-metax cannot run fall back to the `cuda` boxing kernel.
- **`flash_attn`**: Prebuilt MetaX `flash_attn` wheels may ABI-mismatch newer PyTorch versions. Disable or patch before loading Qwen3/transformers if import fails.

### C++ Stub-Only Mode

You can disable the FlagGems Python-layer registration entirely, leaving only the C++ unified wrapper active. This is useful for verifying that all required operators are covered by C++ stubs.

```bash
# Required: tell FlagGems C++ native API where to find Triton kernel sources
export FLAGGEMS_SOURCE_DIR=$(python -c "import os;import flag_gems;print(os.path.dirname(flag_gems.__file__))")

python your_script.py
```

In this mode, all operator dispatch is handled by the C++ dispatch stub (`backends.conf` routing), with no Python-layer `torch.library` registrations from FlagGems.

### Query Status

```python
torch_fl.flagos.is_available()       # Check if device is available
torch_fl.flagos.device_count()       # Number of devices
torch_fl.flagos.current_device()     # Current device index
torch_fl.flagos.synchronize()        # Synchronize device
torch_fl.is_flaggems_enabled()       # Check if FlagGems operators are registered
torch_fl.get_registered_ops()        # List of registered operators
```

## Backend Configuration

You can configure whether to use FlagGems or CUDA backend at per-operator granularity.

### Configuration File

Default path is `torch_fl/configs/backends.conf`, can be overridden via `FLAGOS_BACKEND_CONFIG` environment variable:

```ini
# Format: op_name = backend
# backend: "flagos" | "flaggems" | "cuda"
# Operators not listed default to flagos (FlagGems)
mm = cuda
bmm = flagos
cat = cuda
```

### Environment Variable Override

Individual operators can be overridden via environment variables (higher priority):

```bash
# Format: FLAGOS_OP_<op_name>=cuda|flaggems
# Replace "." in operator names with "__"
export FLAGOS_OP_mm=cuda
export FLAGOS_OP_mm__out=cuda
```

### MetaX backend configs

| File | Purpose |
|------|---------|
| `torch_fl/configs/backends_metax.conf` | All listed ops → `metax` C++ kernels. Default when pytest detects MetaX (`/dev/mxcd`) and `FLAGOS_BACKEND_CONFIG` is unset. |
| `torch_fl/configs/backends_metax_flagos_py.conf` | **Recommended for integration tests.** Hybrid routing: most compute ops → `flagos_python`; keep Triton-incompatible ops (`mm`/`bmm`/`mean.dim`) and factory/allocation ops (`zeros`, `scalar_tensor`, `embedding`, …) on `metax`. |

Example (`backends_metax_flagos_py.conf`):

     # elementwise / inference-path ops
     abs = flagos_python
     add.Tensor = flagos_python
     cos = flagos_python
     sin = flagos_python

     # Triton-incompatible
     mm = metax
     bmm = metax
     mean.dim = metax
     # factory/allocation
     zeros = metax
     scalar_tensor = metax

### Debug Dispatch

```bash
export FLAGOS_LOG_DISPATCH=1  # Print backend selection for each operator dispatch
```

## Testing

Tests in `tests/integration/ops/` are marked with `@pytest.mark` to indicate platform scope:

| Mark | Meaning | When to run |
|------|---------|-------------|
| `@pytest.mark.anyplatform` | Correctness tests, run everywhere | Always |
| `@pytest.mark.cuda` | CUDA/FlagGems dispatch routing tests | CUDA platform only |
| `@pytest.mark.ascend` | Ascend backend dispatch tests | Ascend platform only |

### CUDA Platform

```bash
# Operator tests (requires FlagGems source for C++ native API)
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/ops/ -v -m "anyplatform or cuda"

# Qwen3 inference test
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 training test (single GPU)
FLAGGEMS_SOURCE_DIR=/path_to_repos/FlagGems/src/flag_gems \
  pytest tests/integration/test_qwen3_train.py -v -s --steps 10

# Run only CUDA-specific tests
pytest tests/integration/ops/ -v -m cuda

# Run only FlagGems (Triton) backend tests
pytest tests/integration/ops/ -v -m flaggems

# Run only FlagGems Python wrapper tests
pytest tests/integration/ops/ -v -m flaggems_python

# Run platform-agnostic correctness tests
pytest tests/integration/ops/ -v -m anyplatform

# FlagGems Python wrapper (flagos_python) end-to-end tests
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_flagos_py.conf \
  pytest tests/integration/ops/ -v
```

### MetaX Platform

```bash
# Runtime (see "MetaX Runtime Setup" above)
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:/opt/maca/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/opt/maca/lib/libsymbol_cu.so
export FLAGOS_METAX_CUDART_SHIM=1
export FLAGOS_METAX_COMPAT=1
export GEMS_VENDOR=metax
export FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax_flagos_py.conf
export FLAGGEMS_SOURCE_DIR=$(python -c "import os,flag_gems;print(os.path.dirname(flag_gems.__file__))")

# Basic op tests (includes Qwen3 inference-path ops: cos/sin/rsqrt/silu/...)
pytest tests/integration/test_ops.py -v

# Per-op dispatch tests (hybrid config)
pytest tests/integration/ops/ -v

# Qwen3 inference
pytest tests/integration/test_qwen3_infer.py -v -s --model /path/to/Qwen3-0.6B

# Qwen3 training (single device)
pytest tests/integration/test_qwen3_train.py -v -s --steps 10

# All-metax C++ kernel mode (no flagos_python)
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_metax.conf \
  FLAGOS_DISABLE_FLAGGEMS_PY=1 \
  pytest tests/integration/test_ops.py -v
```

If `FLAGOS_BACKEND_CONFIG` is not set, `tests/integration/conftest.py` auto-selects `torch_fl/configs/backends_metax.conf` on MetaX hardware.

### Ascend Platform

```bash
# Operator tests
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
  pytest tests/integration/ops/ -v -m "anyplatform or ascend"

# Qwen3 inference test
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 training test (single GPU)
FLAGOS_BACKEND_CONFIG=torch_fl/configs/backends_ascend.conf \
  pytest tests/integration/test_qwen3_train.py -v -s --steps 10
```

The `test_qwen3_infer.py` and `test_qwen3_train.py` tests use the same code on all platforms — only the installation method (`ACCELERATOR=ascend pip install -e .`) and runtime environment variables differ.

### Pytest Marks

Operator tests in `tests/integration/ops/` use pytest marks to indicate platform/backend requirements:

| Mark | Description |
|------|-------------|
| `@pytest.mark.anyplatform` | Platform-agnostic correctness tests (shape, dtype, broadcast) |
| `@pytest.mark.cuda` | Requires CUDA backend or CUDA reference comparison |
| `@pytest.mark.flaggems` | Requires FlagGems (Triton) backend |
| `@pytest.mark.flaggems_python` | Requires FlagGems Python wrapper (pybind11 path) |
| `@pytest.mark.ascend` | Requires Ascend NPU backend |

Use `-m <mark>` to run specific test categories. Example: `pytest tests/integration/ops/ -m cuda` runs only CUDA tests.

## Project Structure

```
PyTorch-Plugin-FL/
├── include/                  # Public headers
│   ├── flagos.h              #   Unified runtime API (memory, stream, device)
│   └── macros.h              #   Common macros
├── accelerator/              # Hardware abstraction layer
│   ├── csrc/cuda/            #   CUDA runtime implementation
│   ├── csrc/metax/            #   MetaX cudart shim (symbol version compatibility)
│   └── csrc/ascend/           #   Ascend runtime (ACL-based memory, stream, device)
├── csrc/
│   ├── aten/                 # ATen operator layer
│   │   ├── common.{h,cc}     #   Backend config loading, Backend enum
│   │   ├── dispatcher.h      #   Lightweight op dispatcher (replaces PyTorch DispatchStub)
│   │   ├── device_boxing.h   #   Zero-copy flagos↔CUDA tensor metadata conversion
│   │   ├── register.cc       #   PrivateUse1 dispatch key registration
│   │   ├── {op}.{h,cc}       #   Per-operator stub definitions (add, mm, silu, etc.)
│   │   ├── factory_ops/      #   Basic operators (empty, copy, contiguous, set, fallback)
│   │   ├── functional_ops/   #   Compute operators (mm, bmm, cat, embedding, softmax, etc.)
│   │   └── backends/         #   Backend-specific kernel implementations
│   │       ├── cuda/         #     CUDA kernels (cuBLAS, modified PyTorch kernels)
│   │       ├── flagos/       #     FlagGems C++ native API wrappers
│   │       └── ascend/       #     Ascend kernels (ACL NN API)
│   └── runtime/              # Device runtime
│       ├── device_allocator  #   Device memory allocator
│       ├── host_allocator    #   Pinned memory allocator
│       ├── guard             #   DeviceGuard implementation
│       ├── generator         #   RNG generator
│       ├── hooks             #   Runtime hooks
│       └── accelerator/      #   Hardware abstraction layer
│           ├── cuda/         #     CUDA runtime implementation
│           ├── maca/         #     MACA cudart shim (symbol version compatibility)
│           └── ascend/       #     Ascend runtime (ACL-based memory, stream, device)
├── torch_fl/
│   ├── __init__.py           # Plugin entry point: register device, load FlagGems operators
│   ├── flagos/               # Python device module (stream, event, RNG, AMP)
│   ├── accelerator/          # Python accelerator module (MACA shim loader)
│   ├── backends.conf              # Default backend routing config (CUDA/FlagGems)
│   ├── backends_metax.conf        # MetaX: all listed ops → metax
│   ├── backends_metax_flagos_py.conf  # MetaX hybrid: metax + flagos_python
│   ├── backends_flagos_py.conf    # FlagGems Python wrapper routing
│   ├── backends_ascend.conf       # Ascend backend routing (all ops → ascend)
│   ├── distributed.py        # Distributed training support (DDP patch)
│   ├── integration.py        # FlagGems operator registration logic
│   ├── csrc/                 # C extension (module.cc, stub.c)
│   └── lib/                  # Compiled shared libraries (libtorch_fl.so, libflagos.so)
├── tests/
│   ├── integration/          # Automated integration tests
│   │   ├── ops/              #   Per-operator dispatch tests
│   │   ├── test_qwen3_*.py   #   End-to-end model tests
│   │   └── conftest.py       #   Pytest configuration
│   ├── manual/               # Manual test scripts
│   └── common/               # Test utilities
├── debug/                    # Development notes and debug scripts
├── cmake/                    # CMake modules
├── setup.py                  # CMake build entry point
└── pyproject.toml
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│  Python: import torch_fl                                     │
│  ┌────────────────┐  ┌────────────────────────────┐          │
│  │ torch_fl.flagos│  │ torch_fl.distributed       │          │
│  │ (device API)   │  │ (DDP/FSDP patch)           │          │
│  └────────────────┘  └────────────────────────────┘          │
├──────────────────────────────────────────────────────────────┤
│  PrivateUse1 Dispatch                                        │
│  ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐    │
│  │ FlagGems    │  │ CUDA     │  │ Ascend    │  │ CPU    │    │
│  │ (Triton)    │  │ (native) │  │ (ACL NN)  │  │fallback│    │
│  └─────────────┘  └──────────┘  └───────────┘  └────────┘    │
├──────────────────────────────────────────────────────────────┤
│  C++ Runtime (csrc/)                                         │
│  ┌──────────┐ ┌────────┐ ┌───────┐ ┌───────────┐             │
│  │Allocator │ │ Guard  │ │ RNG   │ │ Hooks     │             │
│  └──────────┘ └────────┘ └───────┘ └───────────┘             │
├──────────────────────────────────────────────────────────────┤
│  Hardware Abstraction (accelerator/)                         │
│  ┌──────────────┐  ┌─────────────────────┐  ┌────────────┐   │
│  │ CUDA Runtime │  │ MetaX cu-bridge+shim │  │ Ascend ACL │   │
│  └──────────────┘  └─────────────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## License

Apache-2.0
