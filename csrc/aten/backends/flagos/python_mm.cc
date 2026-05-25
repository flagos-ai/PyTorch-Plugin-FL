// Copyright (c) 2026, BAAI. All rights reserved.

#include "python_op_caller.h"
#include "../../mm.h"

namespace at::native::flagos {

namespace {

void MmKernelPython(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::Tensor result = CallPythonOp_TT("mm", self, mat2);
  out.copy_(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kFlagOsPython, MmKernelPython)

} // namespace at::native::flagos
