// Copyright (c) 2026, BAAI. All rights reserved.

#include "silu_backward.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(SiluBackwardFn, silu_backward_dispatcher, "silu_backward")

} // namespace at::native::flagos
