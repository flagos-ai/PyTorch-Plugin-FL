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

REGISTER_IMPL_TO_DISPATCHER(SumDimFn, sum_dim_dispatcher, Backend::kFlagOsPython, SumDimKernelPython)

} // namespace at::native::flagos
