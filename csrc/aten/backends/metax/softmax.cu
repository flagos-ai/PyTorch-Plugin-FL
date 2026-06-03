// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../softmax.h"

#include "softmax_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    SoftmaxFn, softmax_stub, FlagosDevice::kMetax, SoftmaxKernelMetax)

}  // namespace at::native::flagos
