// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../silu.h"

namespace at::native::flagos {

namespace {

at::Tensor SiluKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("silu", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SiluFn, silu_stub, FlagosDevice::kFlagOsPython, SiluKernelPython)

} // namespace at::native::flagos
