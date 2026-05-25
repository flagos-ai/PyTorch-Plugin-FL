// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../div_scalar.h"

namespace at::native::flagos {

namespace {

at::Tensor DivScalarKernelPython(const at::Tensor& self, const at::Scalar& other) {
  return CallPythonOp_TS("div", self, other);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(DivScalarFn, div_scalar_stub, FlagosDevice::kFlagOsPython, DivScalarKernelPython)

} // namespace at::native::flagos
