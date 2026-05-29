// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../mm.h"

namespace at::native::flagos {

namespace {

void MmKernelPython(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::Tensor result = CallPythonOp_TT("mm", self, mat2);
  out.copy_(result);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOsPython, MmKernelPython)

} // namespace at::native::flagos
