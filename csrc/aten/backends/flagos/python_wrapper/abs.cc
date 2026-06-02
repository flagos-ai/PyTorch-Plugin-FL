// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../abs.h"

namespace at::native::flagos {

namespace {

at::Tensor AbsKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("abs", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AbsFn, abs_dispatcher, Backend::kFlagOsPython, AbsKernelPython)

} // namespace at::native::flagos
