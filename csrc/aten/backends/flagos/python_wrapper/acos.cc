// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../acos.h"

namespace at::native::flagos {

namespace {

at::Tensor AcosKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("acos", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AcosFn, acos_dispatcher, Backend::kFlagOsPython, AcosKernelPython)

} // namespace at::native::flagos
