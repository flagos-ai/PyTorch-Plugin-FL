// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../argmax.h"
#include "../../device_boxing.h"

#include <ATen/ops/argmax.h>
#include <ATen/ops/argmin.h>

namespace at::native::flagos {

namespace {

at::Tensor ArgmaxKernelCuda(
    const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
  DeviceBoxingGuard guard(self);
  auto result = at::argmax(self, dim, keepdim);
  UnboxToFlagos(result);
  return result;
}

at::Tensor ArgminKernelCuda(
    const at::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
  DeviceBoxingGuard guard(self);
  auto result = at::argmin(self, dim, keepdim);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(ArgmaxFn, argmax_dispatcher, Backend::kCuda, ArgmaxKernelCuda)
REGISTER_IMPL_TO_DISPATCHER(ArgminFn, argmin_dispatcher, Backend::kCuda, ArgminKernelCuda)

} // namespace at::native::flagos
