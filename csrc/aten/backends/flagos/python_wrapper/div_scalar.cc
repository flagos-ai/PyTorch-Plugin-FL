// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../div_scalar.h"

namespace at::native::flagos {

namespace {

at::Tensor DivScalarKernelPython(const at::Tensor& self, const at::Scalar& other) {
  return CallPythonOp_TS("true_divide", self, other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(DivScalarFn, div_scalar_dispatcher, Backend::kFlagOsPython, DivScalarKernelPython)

} // namespace at::native::flagos
