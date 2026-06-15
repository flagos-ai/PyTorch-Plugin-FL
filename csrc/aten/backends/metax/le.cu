// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../le.h"

#include "le_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    LeTensorFn, le_tensor_dispatcher, Backend::kMetax, LeTensorKernel)

} // namespace at::native::flagos
