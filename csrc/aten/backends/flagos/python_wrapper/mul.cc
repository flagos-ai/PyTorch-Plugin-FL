// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../mul.h"

namespace at::native::flagos {

namespace {

at::Tensor MulTensorKernelPython(const at::Tensor& self, const at::Tensor& other) {
  return CallPythonOp_TT("mul", self, other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MulTensorFn, mul_tensor_dispatcher, Backend::kFlagOsPython, MulTensorKernelPython)

} // namespace at::native::flagos
