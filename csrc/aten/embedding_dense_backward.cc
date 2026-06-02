// Copyright (c) 2026, BAAI. All rights reserved.

#include "embedding_dense_backward.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(EmbeddingDenseBackwardFn, embedding_dense_backward_dispatcher, "embedding_dense_backward")

} // namespace at::native::flagos
