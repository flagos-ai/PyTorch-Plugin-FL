// Copyright (c) 2026, BAAI. All rights reserved.

#include "silu.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(SiluFn, silu_dispatcher, "silu")

} // namespace at::native::flagos
