// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../zeros.h"

#include <ATen/ops/empty.h>

namespace at::native::flagos {

namespace {

at::Tensor ZerosKernelCuda(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
    .dtype(dtype.value_or(at::kFloat))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty(size, options);
  result.zero_();
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(ZerosFn, zeros_dispatcher, Backend::kCuda, ZerosKernelCuda)

} // namespace at::native::flagos
