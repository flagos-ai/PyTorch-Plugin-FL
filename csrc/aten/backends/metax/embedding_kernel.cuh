// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t>
__global__ void EmbeddingForwardKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ indices,
    int64_t embed_dim,
    int64_t num_weights,
    int64_t padding_idx) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const int64_t index = indices[idx];
  scalar_t* out_row = out + idx * embed_dim;
  if (padding_idx >= 0 && index == padding_idx) {
    for (int64_t d = 0; d < embed_dim; ++d) {
      out_row[d] = scalar_t(0);
    }
    return;
  }
  if (index < 0 || index >= num_weights) {
    return;
  }
  const scalar_t* weight_row = weight + index * embed_dim;
  for (int64_t d = 0; d < embed_dim; ++d) {
    out_row[d] = weight_row[d];
  }
}

template <typename scalar_t>
void LaunchEmbedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    at::Tensor& output,
    int64_t padding_idx) {
  const at::Tensor weight_c = weight.contiguous();
  const at::Tensor indices_c = indices.contiguous();
  const int64_t num_indices = indices_c.numel();
  const int64_t embed_dim = weight_c.size(1);
  const int64_t num_weights = weight_c.size(0);

  scalar_t* out_ptr = output.data_ptr<scalar_t>();
  const scalar_t* weight_ptr = weight_c.data_ptr<scalar_t>();
  const int64_t* indices_ptr = indices_c.data_ptr<int64_t>();

  metax::Launch1d(
      num_indices,
      EmbeddingForwardKernel<scalar_t>,
      out_ptr,
      weight_ptr,
      indices_ptr,
      embed_dim,
      num_weights,
      padding_idx);
}

}  // namespace

inline at::Tensor EmbeddingKernel(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  TORCH_CHECK(
      !sparse,
      "MetaX embedding: sparse gradients are not supported");
  TORCH_CHECK(
      !scale_grad_by_freq,
      "MetaX embedding: scale_grad_by_freq is not supported");
  TORCH_CHECK(
      weight.dim() == 2,
      "MetaX embedding: weight must be 2-D");
  TORCH_CHECK(
      indices.scalar_type() == at::kLong,
      "MetaX embedding: indices must be int64");

  std::vector<int64_t> out_sizes(indices.sizes().begin(), indices.sizes().end());
  out_sizes.push_back(weight.size(1));
  at::Tensor output = at::empty(out_sizes, weight.options());

  if (output.numel() == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_metax",
      [&]() {
        LaunchEmbedding<scalar_t>(
            weight, indices, output, padding_idx);
      });

  return output;
}

}  // namespace at::native::flagos
