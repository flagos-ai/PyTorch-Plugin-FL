// Copyright (c) 2026, BAAI. All rights reserved.

#include "embedding_dense_backward.h"

#include <ATen/ops/embedding_dense_backward_native.h>
#include "../device_boxing.h"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(EmbeddingDenseBackwardFn, embedding_dense_backward_stub, "embedding_dense_backward")

namespace {

at::Tensor EmbeddingDenseBackwardKernelCuda(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  DeviceBoxingGuard guard(grad_output, indices);
  auto result = at::native::embedding_dense_backward_cuda(
      grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  UnboxToFlagos(result);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(EmbeddingDenseBackwardFn, embedding_dense_backward_stub, FlagosDevice::kCuda, EmbeddingDenseBackwardKernelCuda)

} // namespace at::native::flagos
