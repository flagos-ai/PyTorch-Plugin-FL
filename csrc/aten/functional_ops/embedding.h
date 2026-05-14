// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using EmbeddingFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                                     int64_t, bool, bool);
FLAGOS_DECLARE_DISPATCH(EmbeddingFn, embedding_stub)

} // namespace at::native::flagos
