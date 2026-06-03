// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"

#include "mean_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    MeanDimFn, mean_dim_stub, FlagosDevice::kMetax, MeanDimKernel)

}  // namespace at::native::flagos
