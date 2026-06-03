// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding_dense_backward.h"

#include <ATen/ops/embedding_dense_backward.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingDenseBackwardKernelMetax(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  const at::Tensor grad_cpu = grad_output.cpu();
  const at::Tensor indices_cpu = indices.cpu();
  at::Tensor out_cpu = at::embedding_dense_backward(
      grad_cpu, indices_cpu, num_weights, padding_idx, scale_grad_by_freq);
  return out_cpu.to(grad_output.device(), out_cpu.scalar_type());
}

}  // namespace

FLAGOS_REGISTER_DISPATCH(
    EmbeddingDenseBackwardFn,
    embedding_dense_backward_stub,
    FlagosDevice::kMetax,
    EmbeddingDenseBackwardKernelMetax)

}  // namespace at::native::flagos
