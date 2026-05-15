// Copyright (c) 2026, BAAI. All rights reserved.

#include "nll_loss.h"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(NllLossForwardFn, nll_loss_forward_stub, "nll_loss_forward")
FLAGOS_DEFINE_DISPATCH(NllLossBackwardFn, nll_loss_backward_stub, "nll_loss_backward")

} // namespace at::native::flagos
