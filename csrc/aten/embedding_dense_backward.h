// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using EmbeddingDenseBackwardFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                                                int64_t, int64_t, bool);
FLAGOS_DECLARE_DISPATCH(EmbeddingDenseBackwardFn, embedding_dense_backward_stub)

} // namespace at::native::flagos
