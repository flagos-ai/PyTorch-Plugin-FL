// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../add.h"

namespace at::native::flagos {

namespace {

at::Tensor AddTensorKernelPython(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return CallPythonOp_TTS("add", self, other, alpha);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AddTensorFn, add_tensor_dispatcher, Backend::kFlagOsPython, AddTensorKernelPython)

} // namespace at::native::flagos
