// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../generated/ops.h"

namespace at::native::flagos {

namespace {

// Functional: aten::mm(self, mat2) -> Tensor
at::Tensor MmKernelPython(const at::Tensor& self, const at::Tensor& mat2) {
  return CallPythonOp_TT("mm", self, mat2);
}

// Out variant: aten::mm.out(self, mat2, *, out) -> Tensor&
at::Tensor& MmOutKernelPython(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  out.copy_(CallPythonOp_TT("mm", self, mat2));
  return out;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOsPython, MmKernelPython)
REGISTER_IMPL_TO_DISPATCHER(MmOutFn, mm_out_dispatcher, Backend::kFlagOsPython, MmOutKernelPython)

} // namespace at::native::flagos
