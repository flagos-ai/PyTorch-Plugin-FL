// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../softmax.h"

#include "softmax_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    SoftmaxFn, softmax_dispatcher, Backend::kMetax, SoftmaxKernelMetax)

} // namespace at::native::flagos
