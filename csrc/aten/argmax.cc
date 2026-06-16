// Copyright (c) 2026, BAAI. All rights reserved.

#include "argmax.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(ArgmaxFn, argmax_dispatcher, "argmax")
ADD_IMPL_TO_DISPATCHER(ArgminFn, argmin_dispatcher, "argmin")

} // namespace at::native::flagos
