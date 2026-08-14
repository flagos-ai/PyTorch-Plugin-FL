# torch.compile Integration for flagos Device

This document describes the `torch.compile` integration for the flagos device, enabling automatic kernel fusion and optimization via TorchInductor.

## Overview

The flagos device now supports PyTorch 2.0+ `torch.compile` for automatic performance optimization:

```python
import torch
import torch_fl

model = MyModel().to("flagos:0")
compiled_model = torch.compile(model, backend="flagos")

# Automatic fusion of elementwise ops, reduced dispatch overhead
output = compiled_model(input)
```

**Key benefits**:
- Automatic kernel fusion (no manual optimization needed), cutting per-op dispatch overhead
- Graph stays on flagos: no cuda round trip, no copy at the graph boundary
- Compatible with existing flagos dispatch (FlagGems Python/C++, CUDA boxing)
- Optional FlagTree integration for multi-backend compilation

## Quick Start

### Basic Usage

```python
import torch
import torch_fl

# Standard model definition
model = torch.nn.Sequential(
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
).to("flagos:0")

# Compile with flagos backend
model = torch.compile(model, backend="flagos")

# Use as normal
x = torch.randn(64, 512, device="flagos:0")
y = model(x)  # Automatically fused kernels
```

### Compilation Modes

```python
# Default: balanced optimization
model = torch.compile(model, backend="flagos")

# Maximum performance (longer compile, better runtime)
model = torch.compile(model, backend="flagos", mode="max-autotune")

# Explicit inductor config overrides
model = torch.compile(model, backend="flagos", options={"max_autotune": True})
```

`mode` and `options` are expanded into inductor config patches scoped to that
compile. Note that CUDA graphs are always forced off (see Limitations), so
`mode="reduce-overhead"` -- whose main lever is cudagraphs -- has little effect
here.

### FlagTree Integration (Phase 2)

Use FlagTree for multi-backend kernel compilation:

```bash
# Install FlagTree
pip install flagtree

# Enable FlagTree backend
export FLAGOS_USE_FLAGTREE=1
python your_script.py
```

FlagTree replaces OpenAI Triton with a multi-backend compiler supporting NVIDIA, Ascend, Cambricon, and MetaX hardware.

## Architecture

### Phase 1: Inductor Integration

flagos is registered with TorchInductor as a **first-class GPU device**. The
traced graph is handed to `compile_fx` unchanged -- still on flagos -- and
inductor generates Triton kernels that operate on flagos tensors directly.

**Components**:
1. **Backend registration** (`torch_fl/compile/inductor_backend.py`)
   - Registers `"flagos"` with `torch._dynamo.register_backend`
   - Expands `mode` / `options` into inductor `config_patches`
   - Delegates to `compile_fx` with no graph rewriting

2. **Device interface** (`torch_fl/compile/device_interface.py`)
   - `DeviceInterface` subclass: device state from `torch.flagos`, hardware
     properties from `torch.cuda` (the same physical GPU)
   - Adds `"flagos"` to inductor's `GPU_TYPES` so `is_gpu()` is True and the
     Triton codegen path is taken instead of C++/CPU
   - Reports flagos as cuda at the Triton boundary (`DeviceProperties.create`),
     because Triton's NVIDIA backend hard-checks `target.backend == "cuda"`

3. **Codegen registration** (`torch_fl/compile/inductor_codegen.py`)
   - Device op overrides (guards, streams, sync) inheriting the CUDA ones
   - Scheduling + wrapper codegen: the stock CUDA/Triton pipeline

4. **Dispatch integration**
   - Ops inductor does not fuse fall back to eager flagos dispatch
     (FlagGems Python/C++ or CUDA boxing) with no changes needed

**Flow**:
```
torch.compile(model, backend="flagos")
  → dynamo captures FX graph (on flagos)
  → compile_fx / AOT autograd, graph never leaves flagos
  → inductor generates fused Triton kernels for flagos tensors
  → unfused ops fall back to flagos eager dispatch
```

**Why the graph is not rewritten to cuda.** An earlier version converted the
graph and example inputs to cuda first. Beyond the copy per call, this breaks
backward: `at::getAccelerator()` is PrivateUse1/flagos, and
`torch::autograd::Node::stream()` only yields a stream when a node's input
device type equals the accelerator, so a cuda-rewritten graph produces
stream-less autograd nodes and AOT autograd's backward trace trips
`opt_ready_stream && opt_parent_stream` (engine.cpp:1085).

### Phase 2: FlagTree Integration

**Components**:
1. **Triton import patcher** (`torch_fl/compile/flagtree_shim.py`)
   - Replaces `import triton` with `import flagtree`
   - Activated via `FLAGOS_USE_FLAGTREE=1`

2. **Backend selection**
   - FlagTree backend configured via `GEMS_VENDOR` env var
   - Same Triton kernel code, different backend compiler

**Benefits**:
- Multi-backend: same compiled model runs on NVIDIA/Ascend/Cambricon/MetaX
- Future-proof for non-NVIDIA hardware

## Performance

**Not yet measured.** Correctness is verified (`tests/integration/test_compile.py`);
benchmarking the fusion gain, and comparing it against stock `inductor` on cuda,
is still open work. Structurally the two should land close together -- same
inductor fusion passes, same Triton codegen, and since the graph stays on flagos
there is no per-call copy -- but that is an expectation, not a measurement.

### Benchmarking

```bash
# Run performance benchmark
python tests/perf/bench_compile.py --model=mlp --batch-size=64

# Compare with CUDA baseline
python tests/perf/bench_compile.py --model=transformer --compare-cuda

# Test FlagTree integration
FLAGOS_USE_FLAGTREE=1 python tests/perf/bench_compile.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLAGOS_USE_FLAGTREE` | `0` | Use FlagTree instead of OpenAI Triton |
| `FLAGOS_COMPILE_FALLBACK_EAGER` | `0` | Fall back to eager on compile errors |

Existing dispatch variables (`FLAGOS_USE_FLAGGEMS`, `FLAGOS_BACKEND_CONFIG`) still apply to compiled kernels.

## Troubleshooting

### Compilation Errors

**Symptom**: `torch.compile` raises errors during graph capture or codegen.

**Solutions**:
1. Enable fallback to eager: `FLAGOS_COMPILE_FALLBACK_EAGER=1`
2. Check for unsupported ops (dynamic shapes, custom ops)
3. Verify meta implementations for custom ops

### No Speedup

**Symptom**: Compiled model runs at same speed as eager.

**Possible causes**:
1. Model is compute-bound (large matmuls) — fusion won't help much
2. Compilation didn't fuse ops (check inductor logs)
3. Dispatch overhead is small relative to kernel time

**Debug**: Run with `TORCH_LOGS="+inductor"` to see fusion decisions.

### FlagTree Not Loading

**Symptom**: `FLAGOS_USE_FLAGTREE=1` but warning says "falling back to OpenAI Triton".

**Solutions**:
1. Install FlagTree: `pip install flagtree`
2. Verify import works: `python -c "import flagtree"`
3. Check FlagTree supports your hardware (`GEMS_VENDOR` setting)

## Testing

```bash
# Run integration tests
pytest tests/integration/test_compile.py -v

# Test specific scenarios
pytest tests/integration/test_compile.py::test_basic_compile
pytest tests/integration/test_compile.py::test_compile_backward

# Regression guards for the two codegen fixes this integration required
pytest tests/integration/test_compile.py -k fake_tensor
pytest tests/integration/ops/test_clamp_dispatch.py -v

# Test FlagTree integration (requires flagtree installed)
FLAGOS_USE_FLAGTREE=1 pytest tests/integration/test_compile.py::test_flagtree_integration
```

### Codegen fixes this integration required

Two generated-kernel bugs only surface under compilation, so their regression
tests live alongside it:

- **`detach` re-dispatch.** The generated kernel called `at::detach(self)`, which
  is registered on PrivateUse1 too and so dispatched back into itself. Eager hid
  the recursion because `DeviceBoxingGuard` rewrites self's device metadata
  first; under FakeTensor it cannot, since the Python dispatch key sits *above*
  the backend key. Dynamo traces every `nn.Linear` through detach, so this was a
  stack-overflow segfault at trace time. Fixed by emitting `at::native::detach`
  (`NATIVE_DIRECT_VIEW_OPS` in `scripts/codegen_ops.py`).
- **`optional<Tensor>` boxing in in-place kernels.** `gen_inplace` handed only
  plain `at::Tensor` args to `DeviceBoxingGuard`, so `clamp_.Tensor` passed
  unboxed flagos `min`/`max` into a CUDA `self` and crashed. Fixed by
  materializing each optional into a holder, matching `gen_functional_pure`.

## Limitations

1. **torch >= 2.0 required**: Older PyTorch versions don't have `torch.compile`
2. **Inductor-compatible ops only**: Custom C++ ops may not fuse
3. **Dynamic shapes**: Some models with dynamic shapes may not compile
4. **CUDA graphs off**: `torch.cuda.CUDAGraph` is a dummy class in the CPU torch
   wheel, so `triton.cudagraphs` is forced off even under `mode="max-autotune"`
5. **FlagTree maturity**: Backend support varies by hardware (NVIDIA most mature)

## Roadmap

- [x] Phase 1: Inductor integration (flagos as a first-class GPU device)
- [ ] Phase 2: FlagTree integration — shim exists, not yet exercised end-to-end
- [ ] Benchmark fusion gains vs. stock inductor+triton on cuda
- [ ] Phase 3: FlagGems-aware fusion (recognize pre-optimized patterns)
- [ ] Phase 4: Custom fusion patterns for flagos-specific ops

## See Also

- [PyTorch 2.0 torch.compile documentation](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchInductor overview](https://pytorch.org/docs/stable/torch.compiler_inductor_overview.html)
- [FlagTree repository](https://github.com/flagos-ai/FlagTree)
- [CPU torch + external libtorch_cuda.so](../vendors/cuda/external-libtorch-cuda.md) — why several `torch.cuda` bindings need shimming
- [`torch_fl/compile/README.md`](../../torch_fl/compile/README.md) — the registration surface in detail
