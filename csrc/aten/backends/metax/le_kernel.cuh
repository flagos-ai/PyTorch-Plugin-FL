// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void LeContigKernel(
    int64_t n,
    bool* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] =
      static_cast<scalar_t>(self[idx]) <= static_cast<scalar_t>(other[idx]);
}

template <typename scalar_t>
void LaunchLeIter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  bool* out = static_cast<bool*>(iter.data_ptr(0));
  const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
  const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
  metax::Launch1d(n, LeContigKernel<scalar_t>, out, self, other);
}

}  // namespace

inline at::Tensor LeTensorKernel(
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .add_input(other)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .declare_static_dtype(at::kBool)
                  .build();

  if (!iter.is_contiguous() && iter.numel() > 0) {
    if (self.device() != other.device()) {
      if (self.device().is_cpu() && self.numel() == 1) {
        return LeTensorKernel(self.to(other.device(), other.scalar_type()), other);
      }
      if (other.device().is_cpu() && other.numel() == 1) {
        return LeTensorKernel(self, other.to(self.device(), self.scalar_type()));
      }
    } else {
      const auto shape = iter.output().sizes();
      return LeTensorKernel(
          self.expand(shape).contiguous(),
          other.expand(shape).contiguous());
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND3(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          at::ScalarType::Bool,
          sub_iter.common_dtype(),
          "le_metax",
          [&]() { LaunchLeIter<scalar_t>(sub_iter); });
    }
    return iter.output();
  }

  TORCH_INTERNAL_ASSERT(iter.is_contiguous());

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "le_metax",
      [&]() { LaunchLeIter<scalar_t>(iter); });

  return iter.output();
}

}  // namespace at::native::flagos
