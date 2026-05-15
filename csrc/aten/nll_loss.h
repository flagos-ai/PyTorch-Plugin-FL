// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using NllLossForwardFn = std::tuple<at::Tensor, at::Tensor> (*)(
    const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
    int64_t, int64_t);
FLAGOS_DECLARE_DISPATCH(NllLossForwardFn, nll_loss_forward_stub)

using NllLossBackwardFn = at::Tensor (*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, int64_t, int64_t, const at::Tensor&);
FLAGOS_DECLARE_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub)

} // namespace at::native::flagos
