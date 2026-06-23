// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using TopkFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&, int64_t, int64_t, bool, bool);
DECLARE_DISPATCHER(TopkFn, topk_dispatcher)

} // namespace at::native::flagos
