// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../silu.h"

#include "silu_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(SiluFn, silu_dispatcher, Backend::kMetax, SiluKernelMetax)

} // namespace at::native::flagos
