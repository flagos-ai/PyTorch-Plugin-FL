// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../all.h"

namespace at::native::flagos {

namespace {

at::Tensor AllKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("all", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AllFn, all_dispatcher, Backend::kFlagOsPython, AllKernelPython)

} // namespace at::native::flagos
