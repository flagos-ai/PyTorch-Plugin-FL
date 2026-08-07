"""Platform-specific Python backends.

Each subpackage holds code for one accelerator that needs more than the shared
C++ runtime -- currently only `bpu`, whose acceleration comes from a
torch.compile backend rather than per-op kernels. Nothing is imported here:
importing a subpackage pulls in that platform's toolchain, so `torch_fl`
imports it only when the build targets that platform.
"""
