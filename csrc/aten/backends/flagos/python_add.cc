// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../add.h"

namespace at::native::flagos {

namespace {

at::Tensor AddTensorKernelPython(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  return CallPythonOp_TTS("add", self, other, alpha);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AddTensorFn, add_tensor_stub, FlagosDevice::kFlagOsPython, AddTensorKernelPython)

} // namespace at::native::flagos
