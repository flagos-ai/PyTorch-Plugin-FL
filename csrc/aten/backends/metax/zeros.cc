// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../zeros.h"

#include <ATen/ops/empty.h>
#include <include/flagos.h>

namespace at::native::flagos {

namespace {

at::Tensor ZerosKernelMetax(
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
  at::Tensor result = at::empty(size, options);
  if (result.numel() > 0) {
    const Error_t err = Memset(result.data_ptr(), 0, result.nbytes());
    TORCH_CHECK(err == Success, "MetaX zeros: Memset failed");
  }
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    ZerosFn, zeros_dispatcher, Backend::kMetax, ZerosKernelMetax)

} // namespace at::native::flagos
