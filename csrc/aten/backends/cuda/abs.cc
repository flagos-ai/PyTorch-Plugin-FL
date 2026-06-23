// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../abs.h"
#include "../../device_boxing.h"

#include <ATen/ops/abs.h>

namespace at::native::flagos {

namespace {

at::Tensor AbsKernelCuda(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = at::abs(self);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AbsFn, abs_dispatcher, Backend::kCuda, AbsKernelCuda)

} // namespace at::native::flagos
