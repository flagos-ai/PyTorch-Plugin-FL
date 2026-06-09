// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sum.h"

#include "sum_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    SumDimFn, sum_dim_dispatcher, Backend::kMetax, SumDimKernelMetax)

} // namespace at::native::flagos
