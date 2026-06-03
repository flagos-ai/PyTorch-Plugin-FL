// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <optional>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void FillOnesKernel(
    int64_t n,
    scalar_t* __restrict__ out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(1);
}

template <typename scalar_t>
void LaunchFillOnes(at::Tensor& output) {
  const int64_t n = output.numel();
  scalar_t* out_ptr = output.data_ptr<scalar_t>();
  metax::Launch1d(n, FillOnesKernel<scalar_t>, out_ptr);
}

}  // namespace

inline at::Tensor OnesLikeKernelMetax(
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
  at::Tensor result = at::empty(self.sizes(), options, fmt);
  if (result.numel() == 0) {
    return result;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      result.scalar_type(),
      "ones_like_metax",
      [&]() { LaunchFillOnes<scalar_t>(result); });

  return result;
}

}  // namespace at::native::flagos
