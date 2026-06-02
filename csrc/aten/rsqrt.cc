// Copyright (c) 2026, BAAI. All rights reserved.

#include "rsqrt.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(RsqrtFn, rsqrt_dispatcher, "rsqrt")

} // namespace at::native::flagos
