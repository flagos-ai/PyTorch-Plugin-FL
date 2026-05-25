// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../abs.h"

namespace at::native::flagos {

namespace {

at::Tensor AbsKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("abs", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AbsFn, abs_stub, FlagosDevice::kFlagOsPython, AbsKernelPython)

} // namespace at::native::flagos
