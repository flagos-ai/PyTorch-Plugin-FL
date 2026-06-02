// Copyright (c) 2026, BAAI. All rights reserved.

#include "mul.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(MulTensorFn, mul_tensor_dispatcher, "mul.Tensor")

} // namespace at::native::flagos
