// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../neg.h"

#include "neg_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(NegFn, neg_dispatcher, Backend::kMetax, NegKernel)

} // namespace at::native::flagos
