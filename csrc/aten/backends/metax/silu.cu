// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../silu.h"

#include "silu_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(SiluFn, silu_stub, FlagosDevice::kMetax, SiluKernelMetax)

}  // namespace at::native::flagos
