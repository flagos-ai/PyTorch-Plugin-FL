// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../all.h"

#include "all_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kMaca, AllKernel)

} // namespace at::native::flagos
