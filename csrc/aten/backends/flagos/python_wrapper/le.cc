// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../le.h"

namespace at::native::flagos {

namespace {

at::Tensor LeTensorKernelPython(const at::Tensor& self, const at::Tensor& other) {
  return CallPythonOp_TT("le", self, other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(LeTensorFn, le_tensor_dispatcher, Backend::kFlagOsPython, LeTensorKernelPython)

} // namespace at::native::flagos
