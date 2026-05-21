// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void MulContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = self[idx] && other[idx];
  } else {
    out[idx] = self[idx] * other[idx];
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulSelfScalarKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = self[idx] && static_cast<bool>(other_val);
  } else {
    out[idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[idx]) * other_val);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulOtherScalarKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = static_cast<bool>(self_val) && other[idx];
  } else {
    out[idx] = static_cast<scalar_t>(
        self_val * static_cast<opmath_t>(other[idx]));
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchMulIter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));

  if (iter.is_cpu_scalar(1)) {
    const opmath_t other_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    metax::Launch1d(
        n, MulSelfScalarKernel<scalar_t, opmath_t>, out, self, other_val);
  } else if (iter.is_cpu_scalar(2)) {
    const opmath_t self_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
    metax::Launch1d(
        n, MulOtherScalarKernel<scalar_t, opmath_t>, out, self_val, other);
  } else {
    TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
    metax::Launch1d(n, MulContigKernel<scalar_t>, out, self, other);
  }
}

}  // namespace

inline at::Tensor MulTensorKernel(
    const at::Tensor& self,
    const at::Tensor& other) {
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
        return MulTensorKernel(other, self.to(other.device()));
      }
      if (other.device().is_cpu() && other.numel() == 1) {
        return MulTensorKernel(self, other.to(self.device()));
      }
    } else {
      const auto shape = iter.output().sizes();
      return MulTensorKernel(
          self.expand(shape).contiguous(),
          other.expand(shape).contiguous());
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          at::ScalarType::Bool,
          sub_iter.common_dtype(),
          "mul_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            LaunchMulIter<scalar_t, opmath_t>(sub_iter);
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
      "mul_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        LaunchMulIter<scalar_t, opmath_t>(iter);
      });

  return iter.output();
}

}  // namespace at::native::flagos
