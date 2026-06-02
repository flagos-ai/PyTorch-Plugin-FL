// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using EmbeddingDenseBackwardFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                                                int64_t, int64_t, bool);
DECLARE_DISPATCHER(EmbeddingDenseBackwardFn, embedding_dense_backward_dispatcher)

} // namespace at::native::flagos
