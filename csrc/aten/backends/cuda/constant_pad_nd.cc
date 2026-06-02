// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../constant_pad_nd.h"

#include <ATen/ops/constant_pad_nd_native.h>
#include "../../device_boxing.h"

namespace at::native::flagos {

namespace {

at::Tensor ConstantPadNdKernelCuda(
    const at::Tensor& self, at::IntArrayRef pad, const at::Scalar& value) {
  DeviceBoxingGuard guard(self);
  auto result = at::native::constant_pad_nd(self, pad, value);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(ConstantPadNdFn, constant_pad_nd_dispatcher, Backend::kCuda, ConstantPadNdKernelCuda)

} // namespace at::native::flagos
