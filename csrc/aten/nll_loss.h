// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using NllLossForwardFn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
    int64_t, int64_t);
DECLARE_DISPATCHER(NllLossForwardFn, nll_loss_forward_dispatcher)

using NllLossBackwardFn = at::Tensor (*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, const at::Tensor&);
DECLARE_DISPATCHER(NllLossBackwardFn, nll_loss_backward_dispatcher)

} // namespace at::native::flagos
