// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bitwise_and.h"

#include "bitwise_and_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    BitwiseAndTensorFn,
    bitwise_and_tensor_stub,
    FlagosDevice::kMetax,
    BitwiseAndKernelMetax)

}  // namespace at::native::flagos
