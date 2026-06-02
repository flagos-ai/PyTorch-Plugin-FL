// Copyright (c) 2026, BAAI. All rights reserved.

#include "slice_backward.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(SliceBackwardFn, slice_backward_dispatcher, "slice_backward")

} // namespace at::native::flagos
