// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../acos.h"

namespace at::native::flagos {

namespace {

at::Tensor AcosKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("acos", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AcosFn, acos_stub, FlagosDevice::kFlagOsPython, AcosKernelPython)

} // namespace at::native::flagos
