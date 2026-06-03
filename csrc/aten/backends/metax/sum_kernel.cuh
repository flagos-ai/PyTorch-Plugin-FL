// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>
#include <include/flagos.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

std::vector<int64_t> ParseReductionDims(
    at::OptionalIntArrayRef opt_dims,
    int64_t ndim) {
  if (!opt_dims.has_value() || opt_dims->empty()) {
    std::vector<int64_t> dims(ndim);
    std::iota(dims.begin(), dims.end(), 0);
    return dims;
  }
  std::vector<int64_t> dims;
  dims.reserve(opt_dims->size());
  for (const auto dim : *opt_dims) {
    dims.push_back(at::maybe_wrap_dim(dim, ndim));
  }
  return dims;
}

std::vector<int64_t> ShapeReduce(
    at::IntArrayRef shape,
    at::IntArrayRef dims,
    bool keepdim) {
  std::vector<int64_t> out(shape.begin(), shape.end());
  std::vector<int64_t> sorted_dims(dims.begin(), dims.end());
  std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
  for (int64_t dim : sorted_dims) {
    if (keepdim) {
      out[dim] = 1;
    } else {
      out.erase(out.begin() + dim);
    }
  }
  return out;
}

template <typename scalar_t, typename acc_t, typename out_t>
__global__ void SumAlongDimKernel(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    out_t* __restrict__ out,
    const scalar_t* __restrict__ in) {
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t inner_idx =
      static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (outer_idx >= outer || inner_idx >= inner) {
    return;
  }

  acc_t sum = acc_t(0);
  const int64_t base_in = outer_idx * reduce * inner + inner_idx;
  for (int64_t r = 0; r < reduce; ++r) {
    sum += static_cast<acc_t>(in[base_in + r * inner]);
  }
  out[outer_idx * inner + inner_idx] = static_cast<out_t>(sum);
}

template <typename scalar_t, typename acc_t, typename out_t>
at::Tensor SumAlongDim(
    const at::Tensor& input,
    int64_t dim,
    bool keepdim) {
  const int64_t ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "invalid reduction dim");
  const at::Tensor in = input.contiguous();

  int64_t outer = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer *= in.size(i);
  }
  const int64_t reduce = in.size(dim);
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < ndim; ++i) {
    inner *= in.size(i);
  }

  std::vector<int64_t> out_sizes;
  out_sizes.reserve(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) {
      if (keepdim) {
        out_sizes.push_back(1);
      }
    } else {
      out_sizes.push_back(in.size(i));
    }
  }
  at::Tensor output = at::empty(
      out_sizes,
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

  SumAlongDimKernel<scalar_t, acc_t, out_t>
      <<<blocks, threads, 0, metax::CurrentStream()>>>(
          outer,
          reduce,
          inner,
          output.data_ptr<out_t>(),
          in.data_ptr<scalar_t>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX sum kernel launch failed: ",
      cudaGetErrorString(err));

  return output;
}

template <typename scalar_t, typename acc_t, typename out_t>
at::Tensor SumDimsTyped(
    at::Tensor input,
    std::vector<int64_t> dims,
    bool keepdim) {
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
  for (int64_t dim : dims) {
    input = SumAlongDim<scalar_t, acc_t, out_t>(input, dim, keepdim);
  }
  return input;
}

at::Tensor ZerosLikeShape(
    at::IntArrayRef shape,
    const at::TensorOptions& options) {
  at::Tensor result = at::empty(shape, options);
  if (result.numel() > 0) {
    const Error_t err = Memset(result.data_ptr(), 0, result.nbytes());
    TORCH_CHECK(err == Success, "MetaX sum: Memset failed");
  }
  return result;
}

}  // namespace

inline at::Tensor SumDimKernelMetax(
    const at::Tensor& self,
    at::OptionalIntArrayRef opt_dims,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  auto out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  const bool promote_lowp_to_f32 =
      (self.scalar_type() == at::kHalf ||
       self.scalar_type() == at::kBFloat16) &&
      out_dtype == at::kFloat;

  const std::vector<int64_t> dims = ParseReductionDims(opt_dims, self.dim());

  if (self.numel() == 0) {
    const auto shape = ShapeReduce(self.sizes(), dims, keepdim);
    return ZerosLikeShape(shape, self.options().dtype(out_dtype));
  }

  at::Tensor result;
  if (promote_lowp_to_f32) {
    if (self.scalar_type() == at::kHalf) {
      result = SumDimsTyped<at::Half, float, float>(self, dims, keepdim);
    } else {
      result = SumDimsTyped<at::BFloat16, float, float>(self, dims, keepdim);
    }
    return result;
  }

  if (at::isComplexType(self.scalar_type())) {
    AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "sum_metax", [&]() {
      TORCH_CHECK(
          out_dtype == self.scalar_type(),
          "MetaX sum: complex input does not support dtype argument");
      result = SumDimsTyped<scalar_t, scalar_t, scalar_t>(self, dims, keepdim);
    });
    return result;
  }

  if (self.scalar_type() == at::kHalf) {
    if (out_dtype == at::kHalf) {
      result = SumDimsTyped<at::Half, float, at::Half>(self, dims, keepdim);
    } else {
      TORCH_CHECK(out_dtype == at::kFloat, "MetaX sum: unsupported output dtype");
      result = SumDimsTyped<at::Half, float, float>(self, dims, keepdim);
    }
    return result;
  }

  if (self.scalar_type() == at::kBFloat16) {
    if (out_dtype == at::kBFloat16) {
      result = SumDimsTyped<at::BFloat16, float, at::BFloat16>(
          self, dims, keepdim);
    } else {
      TORCH_CHECK(out_dtype == at::kFloat, "MetaX sum: unsupported output dtype");
      result = SumDimsTyped<at::BFloat16, float, float>(self, dims, keepdim);
    }
    return result;
  }

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "sum_metax", [&]() {
    using acc_t = at::opmath_type<scalar_t>;
    if (out_dtype == self.scalar_type()) {
      result = SumDimsTyped<scalar_t, acc_t, scalar_t>(self, dims, keepdim);
    } else {
      TORCH_CHECK(
          out_dtype == at::kFloat,
          "MetaX sum: unsupported output dtype");
      result = SumDimsTyped<scalar_t, acc_t, float>(self, dims, keepdim);
    }
  });
  return result;
}

}  // namespace at::native::flagos
