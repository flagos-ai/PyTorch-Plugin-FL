# Moore Threads MUSA Installation Guide

Moore Threads MUSA uses native operator kernels calling `libmudnn.so`, not CUDA boxing. The MUSA toolkit ships no CUDA runtime, and there is no vendor dispatch key to box into.

## Prerequisites

- **CPU PyTorch 2.10.x**: `torch==2.10.0` from the upstream CPU index
- **MUSA toolkit**: Moore Threads SDK with `musart` runtime and `mudnn` operator library under `/usr/local/musa`
- **Python**: 3.8 or later
- **Operating System**: Linux

The MUSA toolkit provides:
- `musart`: runtime layer (device/memory/stream)
- `mudnn` (`libmudnn.so`): operator library with category-driven API (Unary/Binary/Reduce/MatMul/Softmax/Convolution)

`mudnn` links against `musart` only and pulls in **no torch symbols at all**, making this backend torch-version-agnostic (tested against both 2.9.1 and 2.10.0 from the same source tree, though only 2.10.0 is officially supported).

## Installation

### 1. Install CPU PyTorch

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Build and install torch_fl

```bash
git clone https://github.com/flagos-ai/PyTorch-Plugin-FL.git
cd PyTorch-Plugin-FL

ACCELERATOR=musa pip install --no-build-isolation -v -e .
```

Build flags:
- `ACCELERATOR=musa`: selects the MUSA build path and enables `MUSA_KERNEL=ON`
- `MUSA_KERNEL=ON`: compiles generated `mudnn` operator kernels (automatic when `ACCELERATOR=musa`)
- `CUDA_KERNEL=OFF`: automatically disabled (the MUSA toolkit exports no CUDA symbols)
- `FLAGGEMS_KERNEL=OFF`: automatically disabled
- `--no-build-isolation`: **required** (without it, pip resolves its own torch into a build overlay, and the extension links against that instead of your installed torch, causing `import torch_fl` to fail with `undefined symbol: c10::ValueError`)

The build runs `scripts/codegen_mudnn.py` to generate kernels. Coverage is **64 generated ops** plus 2 handwritten convolution kernels; everything outside that set reaches the `cpu_fallback`.

### Why no-build-isolation is required

Without `--no-build-isolation`, pip creates a temporary build environment and installs its own copy of torch there. The C++ extension then links against that temporary torch, not the one you installed. When you later `import torch_fl` in your actual environment, the torch C++ object layouts don't match, causing symbol resolution failures.

This constraint is specific to MUSA because `mudnn` has no torch dependency at all. Earlier versions of this backend called `torch_musa`'s flat `at::musa::*` API from `libmusa_python.so`, which links against torch and embeds its C++ object layout—pinning the plugin to one exact torch build (`sizeof(c10::MessageLogger)` changed 408 → 400 between 2.9.1 and 2.10, corrupting the vendor `.so`'s stack). `mudnn` avoids that coupling entirely.

## Verification

### Import order with torch_musa

If the `torch_musa` package is installed alongside `torch_fl` (not recommended, but possible), you must **import `torch_fl` before `torch`**, or export `TORCH_DEVICE_BACKEND_AUTOLOAD=0`:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`torch_musa` registers a `torch.backends` entry point, so a bare `import torch` autoloads it and claims the `PrivateUse1` backend name first. `torch_fl` sets `TORCH_DEVICE_BACKEND_AUTOLOAD=0` internally when imported first, which covers the torch_fl-first order. The other order fails with an explicit message.

**Recommendation**: do not install `torch_musa` when using `torch_fl`. Nothing from `torch_musa` is used.

### Import and device availability

```python
import torch_fl  # Must come before torch if torch_musa is present
import torch

print(f"flagos available: {torch_fl.flagos.is_available()}")
print(f"flagos devices: {torch_fl.flagos.device_count()}")

x = torch.randn(64, 64, device="flagos:0")
y = torch.abs(x)
print(f"abs matches CPU: {torch.allclose(y.cpu(), x.cpu().abs())}")
```

### Runtime backend selection

`torch_fl` installs a `lib/flagos_platform` marker so the runtime picks `backends_musa.conf` automatically. No environment variable override is needed.

## Testing

Run the MUSA dispatch suite:

```bash
pytest tests/integration/ops/test_musa_dispatch.py -v
```

This test file checks that ops route to the `musa` backend, exercises per-op environment overrides, and validates results against CPU references. It replaces the generic per-op tests in `tests/integration/ops/`, which assert `-> cuda` routing that MUSA builds cannot produce (no CUDA boxing kernels are compiled).

Run common operator smoke tests:

```bash
pytest tests/integration/ops/test_common_ops.py -v -s --tb=short
```

**Note**: MUSA has no CI configuration file (`.github/configs/musa.yml` does not exist), so all validation is manual.

## Operator Coverage

### Generated operators (64 ops)

Category-driven codegen via `scripts/codegen_mudnn.py` covers:

- **Unary**: abs, sqrt, rsqrt, exp, log, log2, log10, log1p, sin, cos, acos, atan, tanh, sigmoid, silu, relu, gelu, erf, floor, ceil, sign
- **Binary**: add, mul, sub, div, pow, eq, ne, lt, le, gt, ge, maximum, minimum, logical_and, logical_or, logical_xor
- **Reductions**: sum, mean, max, min, argmax, argmin (with dim support)
- **MatMul**: mm, addmm, bmm, baddbmm
- **Softmax**: softmax, log_softmax
- **Composed from single mode**: neg (MUL by -1), trunc (TRUNCATEDIV by 1), expm1 (EXP then SUB 1)

### Handwritten operators (2 ops)

**Convolution** (`csrc/aten/backends/musa/mudnn_conv.cc`):
- `convolution_overrideable` cannot be left unregistered (ATen's default raises rather than being boxable to CPU)
- `mudnn`'s `Convolution` covers 2 spatial dims only:
  - conv1d runs as 2D conv with unit `H` dim (exact against CPU)
  - conv3d takes the CPU fallback
- Bias is a separate broadcast `Binary::ADD`; `RunFusion` accepts bias only for non-grouped 2D
- Algorithm is chosen by trial and cached per shape (since `GetRecommendForwardAlgorithm` can name one that `Run` then rejects)

**Copy and cast** (`csrc/aten/backends/musa/mudnn_copy.cc`):
- Strided copies and dtype casts use `mudnn`'s `Unary::IDENTITY` / `Unary::CAST`
- Handles both in a single device pass
- Without this, `copy_`/`clone`/`contiguous` would reach the CUDA `DispatchStub` and fail

### CPU fallback

Ops with no `mudnn` mode are deliberately left unregistered, so they reach the `cpu_fallback` and stay correct. Examples:
- `sinh`, `cosh`, `asin`: no `mudnn` mode exists

Registering an op with no kernel behind it would trip the dispatcher's "backend not registered" check.

## Platform-specific Behavior

### Stride and broadcasting support

`mudnn` Tensors carry strides on **both** operands and honor 0-strides. This means:
- Broadcasting is just `expand()` (a view, no copy)
- Non-contiguous inputs are read in place
- No `.contiguous()` materialization needed (contrast with GCU's `topsaten`)

### int64 support

int64 works across Unary/Binary/Reduce/MatMul categories. Unlike GCU's `topsaten` (which has no int64 kernels at all), MUSA only falls back for genuinely unmapped dtypes (complex, quantized).

### TF32 matmul behavior

`mudnn` enables TF32 by default, whereas PyTorch defaults `torch.backends.cuda.matmul.allow_tf32` **off**. The handle is refreshed from torch's flag on every op to match PyTorch semantics. Without this, a 64×64 float `mm` drifts ~2e-2 from CPU.

### Broadcast-reduction SIGFPE workaround

`mudnn` v3300's `Reduce` raises `SIGFPE` (an uncatchable crash, not an error status) when reducing over more than one dim of a tensor that is a broadcast of a single element. Conv bias gradients hit exactly that case, since autograd feeds `ones.expand(...)` into the reduction.

A fully broadcast input is materialized before a multi-dim reduction to avoid this crash.

### Caching allocator

`flagos` keeps its own caching allocator over raw `musaMalloc`. `mudnn` allocates nothing on its own beyond op workspaces, which are served from the same allocator via a `MemoryMaintainer`.

## Limitations

### No continuous validation

MUSA has no CI runner (no `.github/configs/musa.yml` config exists), so all tests are manual. The platform status is **Experimental** (see [`docs/reference/compatibility.md`](../../reference/compatibility.md) line 31).

### Distributed support

Distributed collectives are not validated on MUSA hardware. The architecture recommends the FlagCX-only path unless live evidence proves more. See [`docs/architecture/distributed-flagcx.md`](../../architecture/distributed-flagcx.md).

### Profiler and torch.compile not validated

Neither `torch.compile` nor profiler-tracer parity has been validated on MUSA.

### No FlagGems C++ or Python path

No CUDA boxing kernels or FlagGems dispatch are compiled for MUSA (the toolkit exports no CUDA symbols). The build installs a `lib/flagos_platform` marker so `torch_fl` picks `backends_musa.conf`.

## Build without native kernels

To build the runtime layer only (device/memory/stream support) with no native operator kernels:

```bash
ACCELERATOR=musa MUSA_KERNEL=OFF pip install --no-build-isolation -v -e .
```

All compute ops will fall back to CPU. This mode is useful for testing the runtime layer in isolation.

## Reference Documentation

- [Codegen source](../../../scripts/codegen_mudnn.py): category-driven kernel generation for `mudnn`
- [Compatibility matrix](../../reference/compatibility.md): platform status and limitations
- [Environment variables](../../reference/environment-variables.md): runtime environment variables and backend selection
