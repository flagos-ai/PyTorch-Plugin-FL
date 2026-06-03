// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cmath>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename in_t, typename out_t, typename acc_t>
__global__ void SoftmaxAlongDimKernel(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    out_t* __restrict__ out,
    const in_t* __restrict__ in) {
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t inner_idx =
      static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (outer_idx >= outer || inner_idx >= inner) {
    return;
  }

  const int64_t base = outer_idx * reduce * inner + inner_idx;

  acc_t max_val = -std::numeric_limits<acc_t>::infinity();
  for (int64_t r = 0; r < reduce; ++r) {
    const acc_t v = static_cast<acc_t>(in[base + r * inner]);
    if (v > max_val) {
      max_val = v;
    }
  }

  acc_t sum_exp = acc_t(0);
  for (int64_t r = 0; r < reduce; ++r) {
    sum_exp += ::exp(static_cast<acc_t>(in[base + r * inner]) - max_val);
  }
  const acc_t inv_sum = acc_t(1) / sum_exp;

  for (int64_t r = 0; r < reduce; ++r) {
    const acc_t e = ::exp(static_cast<acc_t>(in[base + r * inner]) - max_val);
    out[base + r * inner] = static_cast<out_t>(e * inv_sum);
  }
}

template <typename in_t, typename out_t, typename acc_t>
at::Tensor SoftmaxTyped(const at::Tensor& self, int64_t dim) {
  const at::Tensor in = self.contiguous();
  const int64_t ndim = in.dim();

  int64_t outer = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer *= in.size(i);
  }
  const int64_t reduce = in.size(dim);
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < ndim; ++i) {
    inner *= in.size(i);
  }

  at::Tensor output = at::empty(
      in.sizes(),
      in.options().dtype(c10::CppTypeToScalarType<out_t>::value));

  if (output.numel() == 0) {
    return output;
  }

  const dim3 blocks(
      static_cast<unsigned int>(outer),
      static_cast<unsigned int>(
          (inner + metax::kBlockSize - 1) / metax::kBlockSize),
      1);
  const dim3 threads(1, metax::kBlockSize, 1);

  SoftmaxAlongDimKernel<in_t, out_t, acc_t>
      <<<blocks, threads, 0, metax::CurrentStream()>>>(
          outer,
          reduce,
          inner,
          output.data_ptr<out_t>(),
          in.data_ptr<in_t>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX softmax kernel launch failed: ",
      cudaGetErrorString(err));
  return output;
}

}  // namespace

inline at::Tensor SoftmaxKernelMetax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {
  TORCH_CHECK(!self.is_complex(), "MetaX softmax does not support complex dtype");
  TORCH_CHECK(self.dim() > 0, "MetaX softmax requires dim on non-scalar tensor");
  dim = at::maybe_wrap_dim(dim, self.dim());

  if (self.scalar_type() == at::kHalf || self.scalar_type() == at::kBFloat16) {
    if (half_to_float) {
      if (self.scalar_type() == at::kHalf) {
        return SoftmaxTyped<at::Half, float, float>(self, dim);
      }
      return SoftmaxTyped<at::BFloat16, float, float>(self, dim);
    }
    if (self.scalar_type() == at::kHalf) {
      return SoftmaxTyped<at::Half, at::Half, float>(self, dim);
    }
    return SoftmaxTyped<at::BFloat16, at::BFloat16, float>(self, dim);
  }

  TORCH_CHECK(
      !half_to_float,
      "MetaX softmax: half_to_float is only valid for Half/BFloat16 input");

  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "softmax_metax", [&]() {
    using acc_t = at::opmath_type<scalar_t>;
    result = SoftmaxTyped<scalar_t, scalar_t, acc_t>(self, dim);
  });
  return result;
}

}  // namespace at::native::flagos
