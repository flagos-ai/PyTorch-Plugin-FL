// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using EmbeddingFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                                     int64_t, bool, bool);
DECLARE_DISPATCHER(EmbeddingFn, embedding_dispatcher)

} // namespace at::native::flagos
