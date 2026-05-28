// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../softmax.h"

namespace at::native::flagos {

namespace {

at::Tensor SoftmaxKernelPython(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return CallPythonOp_TIB("softmax", self, dim, half_to_float);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SoftmaxFn, softmax_stub, FlagosDevice::kFlagOsPython, SoftmaxKernelPython)

} // namespace at::native::flagos
