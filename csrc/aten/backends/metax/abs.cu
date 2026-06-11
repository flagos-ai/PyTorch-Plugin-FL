// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../abs.h"

#include "abs_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(AbsFn, abs_dispatcher, Backend::kMetax, AbsKernel)

} // namespace at::native::flagos
