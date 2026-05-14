// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using SliceBackwardFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef,
                                       int64_t, int64_t, int64_t, int64_t);
FLAGOS_DECLARE_DISPATCH(SliceBackwardFn, slice_backward_stub)

} // namespace at::native::flagos
