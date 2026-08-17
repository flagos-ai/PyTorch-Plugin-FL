# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Inductor-based compile backend for the flagos device.

flagos is registered with inductor as a first-class GPU device (see
device_interface.py and inductor_codegen.py), so the traced graph is handed to
`compile_fx` *as is* -- still on flagos. Inductor generates Triton kernels for
it directly.

Why not rewrite the graph to cuda (as an earlier version did): `at::getAccelerator()`
is PrivateUse1/flagos here, and `torch::autograd::Node::stream()` only yields a
stream when a node's input device type matches the accelerator. A cuda-rewritten
graph therefore produces stream-less autograd nodes, and AOT autograd's backward
trace inside compile_fx trips `opt_ready_stream && opt_parent_stream`
(engine.cpp:1085). Staying on flagos also removes a copy-in/copy-out per call.
"""

import os
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.fx
import torch.cuda


def _patch_cuda_rng_for_cpu_torch():
    """
    Workaround for CPU torch + external libtorch_cuda.so setup.

    dynamo tries to capture torch.cuda.get_rng_state() during tracing, but
    CPU torch doesn't have torch._C._cuda_getDevice() binding. We patch
    torch.cuda to provide stub implementations that prevent the crash.

    Only applied when torch._C lacks CUDA bindings (CPU torch build).
    """
    import torch as torch_module

    if hasattr(torch_module._C, "_cuda_getDevice"):
        return  # Native CUDA torch, no patch needed

    # CPU torch detected - patch CUDA RNG functions
    import torch.cuda as cuda_module

    # Stub implementations that won't be called (dynamo just needs them callable)
    def _stub_get_rng_state(device=None):
        # Return empty tensor as placeholder (dynamo won't execute this)
        return torch_module.tensor([], dtype=torch_module.uint8)

    def _stub_set_rng_state(new_state, device=None):
        pass  # No-op

    cuda_module.get_rng_state = _stub_get_rng_state
    cuda_module.set_rng_state = _stub_set_rng_state


# Apply CPU torch workaround at module load time
_patch_cuda_rng_for_cpu_torch()


def _resolve_config_patches(
    mode: Optional[str],
    options: Optional[Dict[str, Any]],
    dynamic: Optional[bool],
) -> Dict[str, Any]:
    """Turn torch.compile's mode/options into an inductor config patch dict.

    Same expansion `_TorchCompileInductorWrapper` does, plus the flagos-specific
    overrides this build needs. Passing these to compile_fx as `config_patches`
    scopes them to this compile, instead of mutating inductor's global config.
    """
    patches: Dict[str, Any] = {}

    if mode and mode != "default":
        from torch._inductor import list_mode_options

        patches.update(list_mode_options(mode, dynamic))
    if options:
        patches.update({k.replace("-", "_"): v for k, v in options.items()})

    # CUDA graphs need torch.cuda.CUDAGraph, a dummy base class in the CPU torch
    # wheel ("Tried to instantiate dummy base class CUDAGraph"). mode=
    # "max-autotune" turns them on, so force them back off.
    patches["triton.cudagraphs"] = False

    # The static launcher needs torch._C._StaticCudaLauncher, which the CPU
    # torch wheel does not build (we supply libtorch_cuda.so externally). Fall
    # back to the regular Triton launch path.
    if not hasattr(torch._C, "_StaticCudaLauncher"):
        patches["use_static_cuda_launcher"] = False

    return patches


def flagos_compile_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    mode: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    dynamic: Optional[bool] = None,
) -> Callable:
    """
    torch.compile backend for flagos device.

    Registers flagos as an inductor GPU device, then delegates the graph to
    inductor unchanged. Generated Triton kernels run on flagos tensors directly.

    `mode` / `options` arrive as kwargs from dynamo's `_TorchCompileWrapper`
    (torch/__init__.py) whenever torch.compile is called with them on a named
    backend; they are expanded into inductor config patches.

    Usage:
        model = torch.compile(model, backend="flagos")
        # or
        model = torch.compile(model, backend="flagos", mode="max-autotune")

    Environment:
        FLAGOS_USE_FLAGTREE=1 : Require that the active triton be FlagTree
        FLAGOS_COMPILE_FALLBACK_EAGER=1 : Fall back to eager on compile errors
    """
    # Import inductor lazily (not all torch builds have it)
    try:
        from torch._inductor.compile_fx import compile_fx
    except ImportError as e:
        if os.environ.get("FLAGOS_COMPILE_FALLBACK_EAGER", "0") == "1":
            return gm.forward
        raise RuntimeError(
            "torch._inductor not available. Install torch with inductor support "
            "or set FLAGOS_COMPILE_FALLBACK_EAGER=1 to fall back to eager."
        ) from e

    config_patches = _resolve_config_patches(mode, options, dynamic)

    # FlagTree substitutes itself for triton at install time, so if it is
    # installed inductor already compiles with it and there is nothing to switch
    # on here. This only asserts that, so the flag cannot silently no-op.
    if os.environ.get("FLAGOS_USE_FLAGTREE", "0") == "1":
        from torch_fl.compile.flagtree_shim import require_flagtree

        require_flagtree()

    # Make inductor treat flagos as a GPU device. Order matters: is_gpu() must
    # answer True and the device interface must be resolvable before the
    # codegen backend registration reads them.
    from torch_fl.compile.device_interface import register_flagos_device_interface
    from torch_fl.compile.inductor_codegen import (
        publish_codegen_on_device_module,
        register_flagos_codegen,
    )

    register_flagos_device_interface()
    publish_codegen_on_device_module()
    register_flagos_codegen()

    # Hand the graph to inductor untouched -- it is on flagos and stays there.
    try:
        return compile_fx(gm, example_inputs, config_patches=config_patches)
    except Exception as e:
        if os.environ.get("FLAGOS_COMPILE_FALLBACK_EAGER", "0") == "1":
            import warnings

            warnings.warn(f"Inductor compilation failed: {e}. Falling back to eager.")
            return gm.forward
        raise


def register_backend():
    """
    Register the flagos backend with torch._dynamo.

    Called automatically on import torch_fl if torch 2.0+ detected.
    """
    try:
        import torch._dynamo

        torch._dynamo.register_backend(
            name="flagos", compiler_fn=flagos_compile_backend
        )
    except (ImportError, AttributeError):
        # torch._dynamo not available (torch < 2.0)
        pass
