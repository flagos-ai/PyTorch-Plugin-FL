// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void BitwiseAndContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = a[idx] & b[idx];
}

template <typename scalar_t>
void LaunchBitwiseAndContig(
    at::Tensor& out_t,
    const at::Tensor& a_t,
    const at::Tensor& b_t) {
  const int64_t n = out_t.numel();
  scalar_t* out = out_t.data_ptr<scalar_t>();
  const scalar_t* a = a_t.data_ptr<scalar_t>();
  const scalar_t* b = b_t.data_ptr<scalar_t>();
  metax::Launch1d(n, BitwiseAndContigKernel<scalar_t>, out, a, b);
}

}  // namespace

inline at::Tensor BitwiseAndKernelMetax(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto result_type = at::result_type(self, other);
  auto self_cast = self.to(self.device(), result_type).contiguous();
  auto other_cast = other.to(other.device(), result_type).contiguous();

  auto shape = at::infer_size(self_cast.sizes(), other_cast.sizes());
  self_cast = self_cast.expand(shape).contiguous();
  other_cast = other_cast.expand(shape).contiguous();
  at::Tensor output = at::empty(shape, self_cast.options());

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_INTEGRAL_TYPES_AND(
      at::ScalarType::Bool,
      output.scalar_type(),
      "bitwise_and_metax",
      [&]() { LaunchBitwiseAndContig<scalar_t>(output, self_cast, other_cast); });

  return output;
}

}  // namespace at::native::flagos
