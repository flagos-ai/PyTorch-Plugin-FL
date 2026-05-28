// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../cos.h"

namespace at::native::flagos {

namespace {

at::Tensor CosKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("cos", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(CosFn, cos_stub, FlagosDevice::kFlagOsPython, CosKernelPython)

} // namespace at::native::flagos
