// Copyright (c) 2026, BAAI. All rights reserved.

#include "bitwise_and.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(BitwiseAndTensorFn, bitwise_and_tensor_dispatcher, "bitwise_and.Tensor")

} // namespace at::native::flagos
