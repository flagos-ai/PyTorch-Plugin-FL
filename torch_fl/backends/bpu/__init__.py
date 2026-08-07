"""D-Robotics RDK BPU BPU support: a torch.compile backend, not per-op kernels.

The BPU executes whole compiled graphs. An aten subgraph is exported to ONNX,
quantized to int8 (a precondition, not an optimization -- hbdk4 lowers float
conv to the CPU), compiled by hbdk4 into a `.hbm`, and executed through the
Horizon runtime. Ops outside a compilable partition keep running eagerly on the
CPU, so a missing hbdk4 costs performance and never correctness.

Usage:

    import torch, torch_fl
    compiled = torch.compile(model.eval(), backend="bpu")

`register()` is called by `torch_fl` when the build targets bpu, so importing
this package a second time is not required.
"""

from __future__ import annotations

from .backend import bpu_backend, register

__all__ = ["bpu_backend", "register"]
