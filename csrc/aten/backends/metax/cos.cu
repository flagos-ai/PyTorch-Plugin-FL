// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cos.h"

#include "cos_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(CosFn, cos_dispatcher, Backend::kMetax, CosKernelMetax)

} // namespace at::native::flagos
