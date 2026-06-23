// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../acos.h"
#include "../../device_boxing.h"

#include <ATen/ops/acos.h>

namespace at::native::flagos {

namespace {

at::Tensor AcosKernelCuda(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  auto result = at::acos(self);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(AcosFn, acos_dispatcher, Backend::kCuda, AcosKernelCuda)

} // namespace at::native::flagos
