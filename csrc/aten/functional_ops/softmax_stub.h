// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using SoftmaxFn = at::Tensor (*)(const at::Tensor&, int64_t, bool);
FLAGOS_DECLARE_DISPATCH(SoftmaxFn, softmax_stub)

} // namespace at::native::flagos
