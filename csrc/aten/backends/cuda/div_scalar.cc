// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../div_scalar.h"
#include "../../device_boxing.h"

#include <ATen/ops/div.h>

namespace at::native::flagos {

namespace {

at::Tensor DivScalarKernelCuda(const at::Tensor& self, const at::Scalar& other) {
  BoxToCuda(self);
  auto result = at::div(self, other);
  UnboxToFlagos(self);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(DivScalarFn, div_scalar_dispatcher, Backend::kCuda, DivScalarKernelCuda)

} // namespace at::native::flagos
