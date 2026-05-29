// Copyright (c) 2026, BAAI. All rights reserved.

#include "scalar_tensor.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(ScalarTensorFn, scalar_tensor_dispatcher, "scalar_tensor")

} // namespace at::native::flagos
