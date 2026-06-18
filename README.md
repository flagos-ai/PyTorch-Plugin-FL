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
    - MetaX cu-bridge library (required only on MetaX platform)
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

MetaX builds compile device kernels with `mxcc`/`cucc` from `csrc/aten/backends/metax/*.cu` and link them into `libtorch_fl.so`. Runtime goes through MetaX cu-bridge (`runtime/accelerator/metax`); it does **not** use PyTorch's `at::cuda` path.

**Prerequisites**

- MetaX MACA SDK (default `/opt/maca`), with cu-bridge and `mxcc`/`cucc` available
- PyTorch wheel compatible with your MetaX stack (see [Runtime notes](#metax-runtime-notes) below)
- FlagGems 5.0.2+ (optional; required only when routing ops to `flagos_python`)

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

# MetaX SDK paths (adjust if MACA is installed elsewhere)
export METAX_PATH=/opt/maca
export PATH=/opt/maca/tools/cu-bridge/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/tools/cu-bridge/lib:/opt/maca/mxgpu_llvm/lib:$LD_LIBRARY_PATH

ACCELERATOR=metax METAX_KERNEL=ON FLAGGEMS_PYTHON=1 FLAGGEMS_KERNEL=0 CUDA_KERNEL=0 \
  pip install --no-build-isolation -vvv -e .
```

> On MetaX, generic PyPI Triton (`nvidia` backend) cannot JIT kernels for MetaX hardware. Use `torch_fl/backends_metax.conf` or `torch_fl/backends_metax_flagos_py.conf` to route incompatible ops to metax C++ kernels (see [MetaX backend configs](#metax-backend-configs)).

### Build from Source (Ascend Platform)

```bash
# Ensure CANN toolkit is installed and environment is sourced
# (typically: source /usr/local/Ascend/ascend-toolkit/set_env.sh)

ACCELERATOR=ascend FLAGGEMS_KERNEL=0 FLAGGEMS_PYTHON=1 \
  CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

On Ascend, FlagGems C++ kernels and CUDA kernels are disabled. Only the Ascend kernel backend (ACL NN API) is compiled, with FlagGems Python wrappers enabled for ops that route to FlagGems.

### Build Environment Variables

| Variable | Description |
|----------|-------------|
| `ACCELERATOR` | Hardware platform: `cuda` (default), `metax`, or `ascend` |
| `CUDA_HOME` | CUDA toolkit path |
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
export FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax_flagos_py.conf
export FLAGGEMS_SOURCE_DIR=$(python -c "import os,flag_gems;print(os.path.dirname(flag_gems.__file__))")
```

#### MetaX runtime notes

- **PyTorch + Triton stack**: Official `maca-pytorch` images ship `torch+metax` and `triton+metax` (outputs `mcfatbin`). A generic PyTorch wheel plus PyPI Triton uses the NVIDIA backend and will fail with `PTX JIT compilation failed` on MetaX unless affected ops are routed to metax C++ kernels.
- **`flash_attn`**: Prebuilt MetaX `flash_attn` wheels may ABI-mismatch newer PyTorch versions. Disable or patch before loading Qwen3/transformers if import fails.
- **`relu` / `sigmoid`**: Not registered via `m.impl` in the current tree; they fall back to CPU. Do not list them as `metax` in config unless GPU kernels are enabled in `MetaxKernels.cmake`.

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

Default path is `torch_fl/backends.conf`, can be overridden via `FLAGOS_BACKEND_CONFIG` environment variable:

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
| `torch_fl/backends_metax.conf` | All listed ops → `metax` C++ kernels. Default when pytest detects MetaX (`/dev/mxcd`) and `FLAGOS_BACKEND_CONFIG` is unset. |
| `torch_fl/backends_metax_flagos_py.conf` | **Recommended for integration tests.** Hybrid routing: most compute ops → `flagos_python`; keep Triton-incompatible ops (`mm`/`bmm`/`mean.dim`) and factory/allocation ops (`zeros`, `scalar_tensor`, `embedding`, …) on `metax`. |

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
FLAGOS_BACKEND_CONFIG=torch_fl/backends_flagos_py.conf \
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
export FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax_flagos_py.conf
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
FLAGOS_BACKEND_CONFIG=torch_fl/backends_metax.conf \
  FLAGOS_DISABLE_FLAGGEMS_PY=1 \
  pytest tests/integration/test_ops.py -v
```

If `FLAGOS_BACKEND_CONFIG` is not set, `tests/integration/conftest.py` auto-selects `torch_fl/backends_metax.conf` on MetaX hardware.

### Ascend Platform

```bash
# Operator tests
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
  pytest tests/integration/ops/ -v -m "anyplatform or ascend"

# Qwen3 inference test
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
  pytest tests/integration/test_qwen3_infer.py -v -s

# Qwen3 training test (single GPU)
FLAGOS_BACKEND_CONFIG=torch_fl/backends_ascend.conf \
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
