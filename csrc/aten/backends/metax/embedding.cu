// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding.h"

#include "embedding_kernel.cuh"

namespace at::native::flagos {

FLAGOS_REGISTER_DISPATCH(
    EmbeddingFn, embedding_stub, FlagosDevice::kMetax, EmbeddingKernel)

}  // namespace at::native::flagos
