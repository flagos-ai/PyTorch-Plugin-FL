// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using AddTensorFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Scalar&);
DECLARE_DISPATCHER(AddTensorFn, add_tensor_dispatcher)

} // namespace at::native::flagos
