// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

at::Tensor EmbeddingKernelFlaggems(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  return flag_gems::embedding(weight, indices, padding_idx,
                              scale_grad_by_freq, sparse);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(EmbeddingFn, embedding_stub, FlagosDevice::kFlagOs, EmbeddingKernelFlaggems)

} // namespace at::native::flagos
