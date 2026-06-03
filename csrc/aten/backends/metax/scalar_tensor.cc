// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../scalar_tensor.h"

#include <ATen/ops/empty.h>

namespace at::native::flagos {

namespace {

at::Tensor ScalarTensorKernelMetax(
    const at::Scalar& s,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype.value_or(at::ScalarType::Float))
                     .layout(layout.value_or(at::kStrided))
                     .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
                     .pinned_memory(pin_memory.value_or(false));
  auto result = at::empty({}, options);
  result.fill_(s);
  return result;
}

}  // namespace

FLAGOS_REGISTER_DISPATCH(
    ScalarTensorFn, scalar_tensor_stub, FlagosDevice::kMetax, ScalarTensorKernelMetax)

}  // namespace at::native::flagos
