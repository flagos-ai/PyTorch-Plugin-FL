// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../bitwise_and.h"

namespace at::native::flagos {

namespace {

at::Tensor BitwiseAndTensorKernelPython(const at::Tensor& self, const at::Tensor& other) {
  return CallPythonOp_TT("bitwise_and_tensor", self, other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(BitwiseAndTensorFn, bitwise_and_tensor_dispatcher, Backend::kFlagOsPython, BitwiseAndTensorKernelPython)

} // namespace at::native::flagos
