// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../pow.h"

namespace at::native::flagos {

namespace {

at::Tensor PowTensorScalarKernelPython(const at::Tensor& self, const at::Scalar& exp) {
  return CallPythonOp_TS("pow", self, exp);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(PowTensorScalarFn, pow_tensor_scalar_stub, FlagosDevice::kFlagOsPython, PowTensorScalarKernelPython)

} // namespace at::native::flagos
