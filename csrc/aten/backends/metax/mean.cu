// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"

#include "mean_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MeanDimFn, mean_dim_dispatcher, Backend::kMetax, MeanDimKernel)

} // namespace at::native::flagos
