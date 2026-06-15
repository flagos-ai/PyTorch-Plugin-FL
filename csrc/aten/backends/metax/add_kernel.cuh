// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename opmath_t>
__global__ void AddContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      static_cast<opmath_t>(self[idx]) +
      alpha * static_cast<opmath_t>(other[idx]));
}

template <typename scalar_t, typename opmath_t>
__global__ void AddSelfScalarKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      static_cast<opmath_t>(self[idx]) + alpha * other_val);
}

template <typename scalar_t, typename opmath_t>
__global__ void AddOtherScalarKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      self_val + alpha * static_cast<opmath_t>(other[idx]));
}

template <typename scalar_t, typename opmath_t>
void LaunchAddIter(at::TensorIteratorBase& iter, opmath_t alpha) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));

  if (iter.is_cpu_scalar(1)) {
    const opmath_t other_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    metax::Launch1d(
        n,
        AddSelfScalarKernel<scalar_t, opmath_t>,
        out,
        self,
        other_val,
        alpha);
  } else if (iter.is_cpu_scalar(2)) {
    const opmath_t self_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(1));
    metax::Launch1d(
        n,
        AddOtherScalarKernel<scalar_t, opmath_t>,
        out,
        self_val,
        other,
        alpha);
  } else {
    TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
    metax::Launch1d(
        n, AddContigKernel<scalar_t, opmath_t>, out, self, other, alpha);
  }
}

} // namespace

inline at::Tensor AddTensorKernel(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .add_input(other)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();

  if (!iter.is_contiguous() && iter.numel() > 0) {
    if (self.device() != other.device()) {
      if (self.device().is_cpu() && self.numel() == 1) {
        return AddTensorKernel(
            other,
            self.to(other.device(), other.scalar_type()),
            alpha);
      }
      if (other.device().is_cpu() && other.numel() == 1) {
        return AddTensorKernel(
            self,
            other.to(self.device(), self.scalar_type()),
            alpha);
      }
    } else {
      const auto shape = iter.output().sizes();
      return AddTensorKernel(
          self.expand(shape).contiguous(),
          other.expand(shape).contiguous(),
          alpha);
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          at::ScalarType::Bool,
          sub_iter.common_dtype(),
          "add_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            LaunchAddIter<scalar_t, opmath_t>(sub_iter, alpha.to<opmath_t>());
          });
    }
    return iter.output();
  }

  TORCH_INTERNAL_ASSERT(iter.is_contiguous());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "add_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        LaunchAddIter<scalar_t, opmath_t>(iter, alpha.to<opmath_t>());
      });

  return iter.output();
}

} // namespace at::native::flagos
