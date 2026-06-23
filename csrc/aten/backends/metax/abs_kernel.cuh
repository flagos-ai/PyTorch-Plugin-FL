// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cmath>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename opmath_t>
__global__ void AbsContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const opmath_t x = static_cast<opmath_t>(self[idx]);
  out[idx] = static_cast<scalar_t>(std::abs(x));
}

template <typename scalar_t, typename opmath_t>
void LaunchAbsIter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
  metax::Launch1d(n, AbsContigKernel<scalar_t, opmath_t>, out, self);
}

} // namespace

inline at::Tensor AbsKernel(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .build();

  if (!iter.is_contiguous() && iter.numel() > 0) {
    const auto shape = iter.output().sizes();
    return AbsKernel(self.expand(shape).contiguous());
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          sub_iter.common_dtype(),
          "abs_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            LaunchAbsIter<scalar_t, opmath_t>(sub_iter);
          });
    }
    return iter.output();
  }

  TORCH_INTERNAL_ASSERT(iter.is_contiguous());

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "abs_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        LaunchAbsIter<scalar_t, opmath_t>(iter);
      });

  return iter.output();
}

} // namespace at::native::flagos
