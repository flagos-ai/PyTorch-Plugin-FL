// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cos.h"

#include "cos_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(CosFn, cos_stub, FlagosDevice::kMetax, CosKernelMetax)

}  // namespace at::native::flagos
