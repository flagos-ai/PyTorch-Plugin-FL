// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../where.h"

#include "where_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    WhereSelfFn, where_self_stub, FlagosDevice::kMetax, WhereKernelMetax)

}  // namespace at::native::flagos
