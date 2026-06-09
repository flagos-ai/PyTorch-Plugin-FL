// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../sin.h"

#include "sin_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(SinFn, sin_dispatcher, Backend::kMetax, SinKernelMetax)

} // namespace at::native::flagos
