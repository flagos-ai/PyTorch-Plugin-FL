// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/constant_pad_nd.h"

#include <ATen/ops/constant_pad_nd_native.h>
#include "../device_boxing.h"

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

FLAGOS_REGISTER_DISPATCH(ConstantPadNdFn, constant_pad_nd_stub, FlagosDevice::kCuda, ConstantPadNdKernelCuda)

} // namespace at::native::flagos
