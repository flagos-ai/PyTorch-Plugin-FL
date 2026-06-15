// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../where.h"

#include "where_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    WhereSelfFn, where_self_dispatcher, Backend::kMetax, WhereKernelMetax)

} // namespace at::native::flagos
