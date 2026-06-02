// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../where.h"

namespace at::native::flagos {

namespace {

at::Tensor WhereSelfKernelPython(const at::Tensor& condition, const at::Tensor& self, const at::Tensor& other) {
  return CallPythonOp_TTT("where_self_out", condition, self, other);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(WhereSelfFn, where_self_dispatcher, Backend::kFlagOsPython, WhereSelfKernelPython)

} // namespace at::native::flagos
