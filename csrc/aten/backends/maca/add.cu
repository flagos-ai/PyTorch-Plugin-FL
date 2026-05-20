// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"

#include "add_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    AddTensorFn, add_tensor_stub, FlagosDevice::kMaca, AddTensorKernel)

} // namespace at::native::flagos
