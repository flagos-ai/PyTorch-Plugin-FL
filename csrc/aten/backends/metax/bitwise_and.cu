// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bitwise_and.h"

#include "bitwise_and_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    BitwiseAndTensorFn,
    bitwise_and_tensor_dispatcher,
    Backend::kMetax,
    BitwiseAndKernelMetax)

} // namespace at::native::flagos
