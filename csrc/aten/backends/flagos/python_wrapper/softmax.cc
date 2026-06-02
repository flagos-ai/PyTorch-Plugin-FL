// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../softmax.h"

namespace at::native::flagos {

namespace {

at::Tensor SoftmaxKernelPython(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return CallPythonOp_TIB("softmax", self, dim, half_to_float);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SoftmaxFn, softmax_dispatcher, Backend::kFlagOsPython, SoftmaxKernelPython)

} // namespace at::native::flagos
