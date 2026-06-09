// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../pow.h"

#include "pow_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    PowTensorScalarFn,
    pow_tensor_scalar_dispatcher,
    Backend::kMetax,
    PowTensorScalarKernelMetax)

} // namespace at::native::flagos
