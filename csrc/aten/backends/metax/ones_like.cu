// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../ones_like.h"

#include "ones_like_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    OnesLikeFn, ones_like_stub, FlagosDevice::kMetax, OnesLikeKernelMetax)

}  // namespace at::native::flagos
