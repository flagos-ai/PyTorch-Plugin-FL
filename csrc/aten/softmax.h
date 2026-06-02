// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using SoftmaxFn = at::Tensor (*)(const at::Tensor&, int64_t, bool);
DECLARE_DISPATCHER(SoftmaxFn, softmax_dispatcher)

} // namespace at::native::flagos
