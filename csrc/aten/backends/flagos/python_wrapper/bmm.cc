// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../bmm.h"

namespace at::native::flagos {

namespace {

void BmmKernelPython(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::Tensor result = CallPythonOp_TT("bmm", self, mat2);
  out.copy_(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(BmmFn, bmm_stub, FlagosDevice::kFlagOsPython, BmmKernelPython)

} // namespace at::native::flagos
