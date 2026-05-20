// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../le.h"

#include "le_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    LeTensorFn, le_tensor_stub, FlagosDevice::kMaca, LeTensorKernel)

} // namespace at::native::flagos
