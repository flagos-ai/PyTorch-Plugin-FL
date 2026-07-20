// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../generated/ops.h"

namespace at::native::flagos {

namespace {

// Functional: aten::bmm(self, mat2) -> Tensor
at::Tensor BmmKernelPython(const at::Tensor& self, const at::Tensor& mat2) {
  return CallPythonOp_TT("bmm", self, mat2);
}

// Out variant: aten::bmm.out(self, mat2, *, out) -> Tensor&
at::Tensor& BmmOutKernelPython(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  out.copy_(CallPythonOp_TT("bmm", self, mat2));
  return out;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kFlagOsPython, BmmKernelPython)
REGISTER_IMPL_TO_DISPATCHER(BmmOutFn, bmm_out_dispatcher, Backend::kFlagOsPython, BmmOutKernelPython)

} // namespace at::native::flagos
