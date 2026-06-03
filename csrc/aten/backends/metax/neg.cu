// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../neg.h"

#include "neg_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(NegFn, neg_stub, FlagosDevice::kMetax, NegKernel)

}  // namespace at::native::flagos
