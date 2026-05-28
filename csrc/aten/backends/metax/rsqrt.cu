// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../rsqrt.h"

#include "rsqrt_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    RsqrtFn, rsqrt_stub, FlagosDevice::kMetax, RsqrtKernelMetax)

}  // namespace at::native::flagos
