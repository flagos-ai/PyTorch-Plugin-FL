// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"

#include "bmm_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(BmmFn, bmm_stub, FlagosDevice::kMetax, BmmKernelMetax)

}  // namespace at::native::flagos
