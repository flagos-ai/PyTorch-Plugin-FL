// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"

#include "add_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    AddTensorFn, add_tensor_dispatcher, Backend::kMetax, AddTensorKernel)

} // namespace at::native::flagos
