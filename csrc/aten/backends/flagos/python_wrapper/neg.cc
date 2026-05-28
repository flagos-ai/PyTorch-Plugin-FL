// Copyright (c) 2026, BAAI. All rights reserved.
//
// Python wrapper backend implementations for FlagGems ops.
// These register kFlagOsPython kernels that call into flag_gems.ops via pybind11.

#include "../python_op_caller.h"
#include "../../../neg.h"

namespace at::native::flagos {

namespace {

at::Tensor NegKernelPython(const at::Tensor& self) {
  return CallPythonOp_T("neg", self);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(NegFn, neg_stub, FlagosDevice::kFlagOsPython, NegKernelPython)

} // namespace at::native::flagos
