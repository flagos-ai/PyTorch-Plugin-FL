// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mm.h"

#include "mm_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(MmFn, mm_stub, FlagosDevice::kMetax, MmKernelMetax)

}  // namespace at::native::flagos
