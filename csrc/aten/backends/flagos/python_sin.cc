// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../sin.h"

namespace at::native::flagos {

namespace {

at::Tensor SinKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("sin", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SinFn, sin_stub, FlagosDevice::kFlagOsPython, SinKernelPython)

} // namespace at::native::flagos
