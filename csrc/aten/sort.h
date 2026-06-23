// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using SortFn = std::tuple<at::Tensor, at::Tensor> (*)(const at::Tensor&, int64_t, bool);
DECLARE_DISPATCHER(SortFn, sort_dispatcher)

} // namespace at::native::flagos
