// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../pow.h"

namespace at::native::flagos {

namespace {

at::Tensor PowTensorScalarKernelPython(const at::Tensor& self, const at::Scalar& exp) {
  return CallPythonOp_TS("pow_tensor_scalar", self, exp);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(PowTensorScalarFn, pow_tensor_scalar_dispatcher, Backend::kFlagOsPython, PowTensorScalarKernelPython)

} // namespace at::native::flagos
