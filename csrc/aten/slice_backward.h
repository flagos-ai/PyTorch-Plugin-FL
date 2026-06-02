// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using SliceBackwardFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef,
                                       int64_t, int64_t, int64_t, int64_t);
DECLARE_DISPATCHER(SliceBackwardFn, slice_backward_dispatcher)

} // namespace at::native::flagos
