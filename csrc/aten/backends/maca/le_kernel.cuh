// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Exception.h>

#include "maca_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void le_contig_kernel(
    int64_t n,
    bool* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(self[idx]) <= static_cast<scalar_t>(other[idx]);
}

template <typename scalar_t>
void launch_le_iter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  bool* out = static_cast<bool*>(iter.data_ptr(0));
  const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
  const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
  maca::launch_1d(n, le_contig_kernel<scalar_t>, out, self, other);
}

} // namespace

inline at::Tensor LeTensorKernel(
    const at::Tensor& self, const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .add_input(other)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .declare_static_dtype(at::kBool)
                  .build();

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND3(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          at::ScalarType::Bool,
          sub_iter.common_dtype(),
          "le_maca",
          [&]() { launch_le_iter<scalar_t>(sub_iter); });
    }
    return iter.output();
  }

  TORCH_CHECK(iter.is_contiguous(), "MACA le requires contiguous flagos tensors");

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "le_maca",
      [&]() { launch_le_iter<scalar_t>(iter); });

  return iter.output();
}

} // namespace at::native::flagos
