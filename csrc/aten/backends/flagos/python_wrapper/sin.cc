// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../sin.h"

namespace at::native::flagos {

namespace {

at::Tensor SinKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("sin", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SinFn, sin_dispatcher, Backend::kFlagOsPython, SinKernelPython)

} // namespace at::native::flagos
