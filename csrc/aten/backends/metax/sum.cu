// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sum.h"

#include "sum_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    SumDimFn, sum_dim_stub, FlagosDevice::kMetax, SumDimKernelMetax)

}  // namespace at::native::flagos
