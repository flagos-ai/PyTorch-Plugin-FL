// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../sum.h"

namespace at::native::flagos {

namespace {

at::Tensor SumDimKernelPython(const at::Tensor& self, at::OptionalIntArrayRef dim,
                              bool keepdim, std::optional<at::ScalarType> dtype) {
  return CallPythonOp_TOIB("sum_dim", self, dim, keepdim, dtype);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SumDimFn, sum_dim_stub, FlagosDevice::kFlagOsPython, SumDimKernelPython)

} // namespace at::native::flagos
