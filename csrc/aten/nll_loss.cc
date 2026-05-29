// Copyright (c) 2026, BAAI. All rights reserved.

#include "nll_loss.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(NllLossForwardFn, nll_loss_forward_dispatcher, "nll_loss_forward")
ADD_IMPL_TO_DISPATCHER(NllLossBackwardFn, nll_loss_backward_dispatcher, "nll_loss_backward")

} // namespace at::native::flagos
