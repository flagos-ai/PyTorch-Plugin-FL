// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding.h"
#include "../../device_boxing.h"

#include <ATen/ops/embedding.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingKernelCuda(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  // Box tensors to CUDA so the composite decomposition (index_select)
  // dispatches to the CUDA kernel instead of falling through to CPU fallback.
  DeviceBoxingGuard guard(weight, indices);
  auto result = at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(EmbeddingFn, embedding_dispatcher, Backend::kCuda, EmbeddingKernelCuda)

} // namespace at::native::flagos
