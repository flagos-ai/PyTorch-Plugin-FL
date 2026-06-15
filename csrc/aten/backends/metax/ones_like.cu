// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../ones_like.h"

#include "ones_like_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    OnesLikeFn, ones_like_dispatcher, Backend::kMetax, OnesLikeKernelMetax)

} // namespace at::native::flagos
