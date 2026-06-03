// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void WhereContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const bool* __restrict__ cond,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = cond[idx] ? self[idx] : other[idx];
}

template <typename scalar_t>
void LaunchWhereContig(
    at::Tensor& out_t,
    const at::Tensor& cond_t,
    const at::Tensor& self_t,
    const at::Tensor& other_t) {
  const int64_t n = out_t.numel();
  scalar_t* out = out_t.data_ptr<scalar_t>();
  const bool* cond = cond_t.data_ptr<bool>();
  const scalar_t* self = self_t.data_ptr<scalar_t>();
  const scalar_t* other = other_t.data_ptr<scalar_t>();
  metax::Launch1d(n, WhereContigKernel<scalar_t>, out, cond, self, other);
}

}  // namespace

inline at::Tensor WhereKernelMetax(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {
  auto result_type = at::result_type(self, other);
  auto cond_bool = condition.scalar_type() == at::kBool
      ? condition
      : condition.to(condition.device(), at::kBool);
  auto self_cast = self.to(self.device(), result_type).contiguous();
  auto other_cast = other.to(other.device(), result_type).contiguous();

  auto shape = at::infer_size(
      cond_bool.sizes(),
      at::infer_size(self_cast.sizes(), other_cast.sizes()));
  cond_bool = cond_bool.expand(shape).contiguous();
  self_cast = self_cast.expand(shape).contiguous();
  other_cast = other_cast.expand(shape).contiguous();
  at::Tensor output = at::empty(shape, self_cast.options());

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      output.scalar_type(),
      "where_metax",
      [&]() {
        LaunchWhereContig<scalar_t>(output, cond_bool, self_cast, other_cast);
      });
  return output;
}

}  // namespace at::native::flagos
