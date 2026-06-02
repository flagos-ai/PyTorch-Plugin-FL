// Copyright (c) 2026, BAAI. All rights reserved.

#include "softmax.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(SoftmaxFn, softmax_dispatcher, "_softmax")

} // namespace at::native::flagos
