// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../silu_backward.h"

namespace at::native::flagos {

namespace {

at::Tensor SiluBackwardKernelPython(const at::Tensor& grad_output, const at::Tensor& self) {
  return CallPythonOp_TT("silu_backward", grad_output, self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SiluBackwardFn, silu_backward_stub, FlagosDevice::kFlagOsPython, SiluBackwardKernelPython)

} // namespace at::native::flagos
