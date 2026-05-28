// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../rsqrt.h"

namespace at::native::flagos {

namespace {

at::Tensor RsqrtKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("rsqrt", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(RsqrtFn, rsqrt_stub, FlagosDevice::kFlagOsPython, RsqrtKernelPython)

} // namespace at::native::flagos
