// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../all.h"

#include "all_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(AllFn, all_dispatcher, Backend::kMetax, AllKernel)

} // namespace at::native::flagos
