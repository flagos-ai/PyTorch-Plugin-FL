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
__global__ void RsqrtContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      opmath_t(1) / ::sqrt(static_cast<opmath_t>(self[idx])));
}

template <typename scalar_t, typename opmath_t>
void LaunchRsqrtIter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
  metax::Launch1d(
      n, RsqrtContigKernel<scalar_t, opmath_t>, out, self);
}

} // namespace

inline at::Tensor RsqrtKernelMetax(const at::Tensor& self) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .build();

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::BFloat16,
          at::ScalarType::Half,
          sub_iter.common_dtype(),
          "rsqrt_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            LaunchRsqrtIter<scalar_t, opmath_t>(sub_iter);
          });
    }
    return iter.output();
  }

  TORCH_CHECK(
      iter.is_contiguous(),
      "MetaX rsqrt requires contiguous flagos tensors");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "rsqrt_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        LaunchRsqrtIter<scalar_t, opmath_t>(iter);
      });

  return iter.output();
}

} // namespace at::native::flagos
