// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../generated/ops.h"

namespace at::native::flagos {

namespace {

at::Tensor SiluKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("silu", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SiluFn, silu_dispatcher, Backend::kFlagOsPython, SiluKernelPython)

} // namespace at::native::flagos
