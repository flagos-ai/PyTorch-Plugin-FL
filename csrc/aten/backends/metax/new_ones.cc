// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../new_ones.h"

#include "../../ones_like.h"
#include <ATen/ops/empty.h>

namespace at::native::flagos {

namespace {

at::Tensor NewOnesKernelMetax(
    const at::Tensor& self,
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(self.scalar_type()))
                     .layout(layout.value_or(self.layout()))
                     .device(device.value_or(self.device()))
                     .pinned_memory(pin_memory.value_or(false));
  at::Tensor result = at::empty(size, options);
  return at::native::flagos::ones_like_stub(
      result,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt);
}

}  // namespace

FLAGOS_REGISTER_DISPATCH(
    NewOnesFn, new_ones_stub, FlagosDevice::kMetax, NewOnesKernelMetax)

}  // namespace at::native::flagos
