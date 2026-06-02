// Copyright (c) 2026, BAAI. All rights reserved.

#include "add.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(AddTensorFn, add_tensor_dispatcher, "add.Tensor")

} // namespace at::native::flagos
