// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../all.h"

namespace at::native::flagos {

namespace {

at::Tensor AllKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("all", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kFlagOsPython, AllKernelPython)

} // namespace at::native::flagos
