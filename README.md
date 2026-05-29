# torch_fl

A custom PyTorch device plugin based on the PrivateUse1 extension mechanism, registering [FlagGems](https://github.com/FlagOpen/FlagGems) high-performance Triton operators as the `flagos` device backend for unified multi-chip support.

## Features

- Automatically registers FlagGems Triton operators as dispatch implementations for the `flagos` device
- Configurable backend routing: select FlagGems or native vendor backend (CUDA/MACA/Ascend) at per-operator granularity
- Currently supports CUDA, MACA (MetaX), and Ascend hardware platforms
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
    - MACA cu-bridge library (required only on MACA platform)
    - CANN toolkit (required only on Ascend platform)
- PyTorch 2.11.0
- FlagGems (version 5.0.2 or higher, requires DFLAGGEMS_BUILD_C_EXTENSIONS enabled). For source installation, refer to: [FlagGems Installation](https://flagos-ai.github.io/FlagGems/getting-started/install/)
  - Note: FlagGems is optional on Ascend platform

### Build from Source (CUDA Platform)

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git && cd PyTorch-Plugin-FL

pip install -e . --no-build-isolation
```

### Build from Source (MACA Platform)

```bash
# Set MACA cu-bridge library path, depending on the actual cu-bridge path in your environment
export LD_LIBRARY_PATH=/opt/maca/tools/cu-bridge/lib:$LD_LIBRARY_PATH

ACCELERATOR=maca pip install -e . --no-build-isolation
```

### Build from Source (Ascend Platform)

```bash
# Ensure CANN toolkit is installed and environment is sourced
# (typically: source /usr/local/Ascend/ascend-toolkit/set_env.sh)

ACCELERATOR=ascend FLAGGEMS_KERNEL=0 CUDA_KERNEL=0 ASCEND_KERNEL=1 \
  pip install --no-build-isolation -vvv -e .
```

On Ascend, FlagGems and CUDA kernels are disabled. Only the Ascend kernel backend (ACL NN API) is compiled.

### Build Environment Variables

| Variable | Description |
|----------|-------------|
| `ACCELERATOR` | Hardware platform: `cuda` (default), `maca`, or `ascend` |
| `CUDA_HOME` | CUDA toolkit path |
| `MACA_PATH` | MACA SDK path (default `/opt/maca`) |
| `ASCEND_HOME` | CANN toolkit path (default `/usr/local/Ascend/ascend-toolkit/latest`) |
| `FLAGGEMS_DIR` | FlagGems C++ library path (enables low-overhead C++ dispatch) |
| `FLAGGEMS_KERNEL` | Enable FlagGems kernel build (`ON`/`OFF`, default `ON`; set `0` for Ascend) |
| `CUDA_KERNEL` | Enable CUDA kernel build (`ON`/`OFF`, default `ON`; set `0` for Ascend) |
| `ASCEND_KERNEL` | Enable Ascend kernel build (`ON`/`OFF`, default `OFF`; set `1` for Ascend) |

### Runtime Environment Variables

| Variable | Description |
|----------|-------------|
| `FLAGGEMS_SOURCE_DIR` | FlagGems source directory (required when C++ native API ops route to `flaggems` backend) |
| `FLAGOS_BACKEND_CONFIG` | Override path for `backends.conf` (use `torch_fl/backends_ascend.conf` on Ascend) |
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

### MACA Platform Import Order

On MetaX (MACA) hardware, you **must** import `torch_fl` before `import torch`:

```python
import torch_fl  # Must import first
import torch
```

Reason: PyTorch's bundled CUDA 12.x runtime is ABI-incompatible with MACA's cu-bridge (CUDA 11.6 compatibility layer). `torch_fl` preloads a shim library to provide the required symbol versions.

This restriction does not apply to CUDA platforms.

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
│   ├── csrc/maca/            #   MACA cudart shim (symbol version compatibility)
│   └── csrc/ascend/          #   Ascend runtime (ACL-based memory, stream, device)
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
│   ├── backends.conf         # Default backend routing config (CUDA/FlagGems)
│   ├── backends_ascend.conf  # Ascend backend routing config (all ops → ascend)
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
│  │ CUDA Runtime │  │ MACA cu-bridge+shim │  │ Ascend ACL │   │
│  └──────────────┘  └─────────────────────┘  └────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## License

Apache-2.0
