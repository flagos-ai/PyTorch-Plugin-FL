// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../embedding.h"

#include "embedding_kernel.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    EmbeddingFn, embedding_dispatcher, Backend::kMetax, EmbeddingKernel)

} // namespace at::native::flagos
