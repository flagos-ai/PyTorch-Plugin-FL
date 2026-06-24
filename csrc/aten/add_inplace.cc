// Copyright (c) 2026, BAAI. All rights reserved.

#include "add_inplace.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(AddInplaceTensorFn, add_inplace_tensor_dispatcher, "add_.Tensor")

} // namespace at::native::flagos
