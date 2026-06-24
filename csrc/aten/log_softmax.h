// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using LogSoftmaxFn = at::Tensor (*)(const at::Tensor&, int64_t, bool);
DECLARE_DISPATCHER(LogSoftmaxFn, log_softmax_dispatcher)

using LogSoftmaxBackwardFn = at::Tensor (*)(
    const at::Tensor&, const at::Tensor&, int64_t, at::ScalarType);
DECLARE_DISPATCHER(LogSoftmaxBackwardFn, log_softmax_backward_dispatcher)

using SoftmaxBackwardFn = at::Tensor (*)(
    const at::Tensor&, const at::Tensor&, int64_t, at::ScalarType);
DECLARE_DISPATCHER(SoftmaxBackwardFn, softmax_backward_dispatcher)

} // namespace at::native::flagos
