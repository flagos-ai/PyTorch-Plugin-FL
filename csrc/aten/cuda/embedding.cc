// Copyright (c) 2026, BAAI. All rights reserved.

#include "../functional/embedding.h"

#include <ATen/ops/embedding_compositeexplicitautograd_dispatch.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingKernelCuda(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  // Fall through to composite decomposition (index_select based)
  return at::compositeexplicitautograd::embedding(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(EmbeddingFn, embedding_stub, FlagosDevice::kCuda, EmbeddingKernelCuda)

} // namespace at::native::flagos
