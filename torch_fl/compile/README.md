# torch.compile Integration for FlagOS

This directory contains the torch.compile backend integration for the flagos device.

## Overview

The `flagos` backend registers flagos with TorchInductor as a **first-class GPU
device**, then hands the traced graph to `compile_fx` unchanged. Inductor
generates Triton kernels that operate on flagos tensors directly -- there is no
conversion to cuda and no copy at the graph boundary.

This works because flagos runs on the physical GPU that `torch.cuda` describes:
its allocator delegates to `c10::cuda::CUDACachingAllocator`, so a flagos
tensor's storage already *is* CUDA memory, and device indices line up
(`flagos.set_device(i)` moves the CUDA current device).

## Usage

```python
import torch
import torch_fl

def my_model(x):
    z = x + 1.0
    z = torch.nn.functional.relu(z)
    z = z * 2.0
    return z

x = torch.randn(4096, 4096, device='flagos:0')

# Compile with flagos backend
compiled_model = torch.compile(my_model, backend='flagos')
result = compiled_model(x)

# mode / options are forwarded to inductor as config patches
compiled_model = torch.compile(my_model, backend='flagos', mode='max-autotune')
```

## Implementation Notes

### Why the graph stays on flagos

An earlier version rewrote the graph and its example inputs to cuda before
calling `compile_fx`. That is not merely a copy cost -- it breaks backward.
`at::getAccelerator()` is PrivateUse1/flagos in this build, and
`torch::autograd::Node::stream()` only yields a stream when a node's input
device type equals the accelerator. A cuda-rewritten graph therefore produces
stream-less autograd nodes, and AOT autograd's backward trace inside
`compile_fx` trips `opt_ready_stream && opt_parent_stream` (engine.cpp:1085).

Keeping the graph on flagos avoids that, and removes a copy-in/copy-out per call.

### Registration surface (`device_interface.py`, `inductor_codegen.py`)

| What | Why |
|---|---|
| `GPU_TYPES.append("flagos")` | `is_gpu()` is a membership test on this list; without it inductor picks the C++/CPU codegen path and never emits Triton. Must be in place -- callers captured the list object at import. |
| Prime `get_gpu_type()`'s cache | It asserts at most one GPU type is available, and the torch.cuda shim reports available alongside flagos. |
| `register_interface_for_device` | Inductor's `DeviceInterface`: device state from `torch.flagos`, hardware properties from `torch.cuda` (same GPU). |
| `DeviceProperties.create` wrap | Reports flagos as cuda at the Triton boundary. Triton's NVIDIA backend hard-checks `target.backend == "cuda"`, so a literal `"flagos"` finds zero compatible backends. Inductor already does this in the opposite direction for ROCm (`hints.py:149`). |
| `register_device_op_overrides` | Device guard / stream / synchronize snippets spliced into generated code. Inherits the CUDA ones; only Python-level device manipulation routes through `torch.flagos`. |
| `register_backend_for_device` | Scheduling + wrapper codegen -- the stock CUDA/Triton pipeline under the `"flagos"` key. |

The four codegen classes are also published on `torch.flagos` so inductor's
official PrivateUse1 hook (`init_backend_registration`, `codegen/common.py:578`)
can register flagos on its own.

### CPU-torch wheel accommodations

This build pairs a CPU-only pip torch with an externally supplied
`libtorch_cuda.so`, so several `torch.cuda` Python bindings are missing. The
backend compensates:

- `use_static_cuda_launcher = False` -- `torch._C._StaticCudaLauncher` is not built.
- `triton.cudagraphs = False` -- `torch.cuda.CUDAGraph` is a dummy base class
  that raises on construction; `mode="max-autotune"` would otherwise enable it.
- `CudaInterface.get_raw_stream` is re-attached -- the binding exists, but the
  import-time `torch.cuda._is_compiled()` probe left it at `None`.

See `torch_fl/accelerator/cuda/_cuda_compat.py` for the memory-stats and
Event/Stream shims that inductor's autotuner needs.

## Environment Variables

- `FLAGOS_USE_FLAGTREE=1` - Reserved for FlagTree integration (Phase 2)
- `FLAGOS_COMPILE_FALLBACK_EAGER=1` - Fall back to eager mode on compile errors

## Limitations

1. Single device - multi-GPU compilation not yet exercised
2. FlagTree integration is a stub (Phase 2)

## Future Work

- [ ] FlagTree integration to replace OpenAI Triton
- [ ] Benchmark fusion gains against stock inductor+triton on cuda
- [ ] Multi-GPU compilation support
