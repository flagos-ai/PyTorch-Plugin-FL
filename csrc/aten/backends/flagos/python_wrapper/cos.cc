// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../cos.h"

namespace at::native::flagos {

namespace {

at::Tensor CosKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("cos", self);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CosFn, cos_dispatcher, Backend::kFlagOsPython, CosKernelPython)

} // namespace at::native::flagos
