// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mul.h"

#include "mul_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    MulTensorFn, mul_tensor_stub, FlagosDevice::kMetax, MulTensorKernel)

}  // namespace at::native::flagos
