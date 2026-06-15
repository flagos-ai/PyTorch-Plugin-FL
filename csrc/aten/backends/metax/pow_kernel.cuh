// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cmath>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename opmath_t>
__global__ void PowContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t exp) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      ::pow(static_cast<opmath_t>(self[idx]), exp));
}

template <typename scalar_t, typename opmath_t>
__global__ void PowSquareContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const opmath_t base = static_cast<opmath_t>(self[idx]);
  out[idx] = static_cast<scalar_t>(base * base);
}

template <typename scalar_t, typename opmath_t>
__global__ void PowCubeContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const opmath_t base = static_cast<opmath_t>(self[idx]);
  out[idx] = static_cast<scalar_t>(base * base * base);
}

template <typename scalar_t, typename opmath_t>
void LaunchPowIter(at::TensorIteratorBase& iter, opmath_t exp) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
  const double d_exp = static_cast<double>(exp);

  if (d_exp == 2.0) {
    metax::Launch1d(
        n, PowSquareContigKernel<scalar_t, opmath_t>, out, self);
  } else if (d_exp == 3.0) {
    metax::Launch1d(
        n, PowCubeContigKernel<scalar_t, opmath_t>, out, self);
  } else {
    metax::Launch1d(
        n, PowContigKernel<scalar_t, opmath_t>, out, self, exp);
  }
}

} // namespace

inline at::Tensor PowTensorScalarKernelMetax(
    const at::Tensor& self,
    const at::Scalar& exp_scalar) {
  const auto common_dtype = at::result_type(self, exp_scalar);
  const at::Tensor base =
      self.scalar_type() == common_dtype ? self : self.to(common_dtype);

  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(base)
                  .build();

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          sub_iter.common_dtype(),
          "pow_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            const auto exp = exp_scalar.to<scalar_t>();
            LaunchPowIter<scalar_t, opmath_t>(
                sub_iter, static_cast<opmath_t>(exp));
          });
    }
    return iter.output();
  }

  TORCH_CHECK(
      iter.is_contiguous(),
      "MetaX pow.Tensor_Scalar requires contiguous flagos tensors");

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "pow_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto exp = exp_scalar.to<scalar_t>();
        LaunchPowIter<scalar_t, opmath_t>(
            iter, static_cast<opmath_t>(exp));
      });

  return iter.output();
}

} // namespace at::native::flagos
