// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/ones_like.h"

#include <ATen/ops/empty.h>

namespace at::native::flagos {

namespace {

at::Tensor OnesLikeKernelCuda(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(self.scalar_type()))
    .layout(layout.value_or(self.layout()))
    .device(device.value_or(self.device()))
    .pinned_memory(pin_memory.value_or(false));
  auto fmt = memory_format.value_or(at::MemoryFormat::Contiguous);
  if (fmt == at::MemoryFormat::Preserve) {
    fmt = self.suggest_memory_format();
  }
  auto result = at::empty(self.sizes(), options, fmt);
  result.fill_(1);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(OnesLikeFn, ones_like_stub, FlagosDevice::kCuda, OnesLikeKernelCuda)

} // namespace at::native::flagos
