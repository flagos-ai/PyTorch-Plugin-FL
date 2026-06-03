// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sin.h"

#include "sin_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(SinFn, sin_stub, FlagosDevice::kMetax, SinKernelMetax)

}  // namespace at::native::flagos
