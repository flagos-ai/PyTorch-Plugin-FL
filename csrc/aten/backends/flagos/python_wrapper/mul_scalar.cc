// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../mul_scalar.h"

namespace at::native::flagos {

namespace {

at::Tensor MulScalarKernelPython(const at::Tensor& self, const at::Scalar& other) {
  return CallPythonOp_TS("mul", self, other);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MulScalarFn, mul_scalar_stub, FlagosDevice::kFlagOsPython, MulScalarKernelPython)

} // namespace at::native::flagos
