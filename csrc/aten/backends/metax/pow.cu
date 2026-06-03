// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../pow.h"

#include "pow_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    PowTensorScalarFn,
    pow_tensor_scalar_stub,
    FlagosDevice::kMetax,
    PowTensorScalarKernelMetax)

}  // namespace at::native::flagos
