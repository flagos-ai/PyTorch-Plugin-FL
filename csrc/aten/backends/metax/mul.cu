// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mul.h"

#include "mul_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MulTensorFn, mul_tensor_dispatcher, Backend::kMetax, MulTensorKernel)

} // namespace at::native::flagos
