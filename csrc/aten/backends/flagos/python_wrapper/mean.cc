// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../generated/ops.h"

namespace at::native::flagos {

namespace {

at::Tensor MeanDimKernelPython(const at::Tensor& self, at::OptionalIntArrayRef dim,
                               bool keepdim, std::optional<at::ScalarType> dtype) {
  return CallPythonOp_TOIB("mean_dim", self, dim, keepdim, dtype);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MeanDimFn, mean_dim_dispatcher, Backend::kFlagOsPython, MeanDimKernelPython)

} // namespace at::native::flagos
