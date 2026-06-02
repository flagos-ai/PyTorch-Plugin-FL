// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using BitwiseAndTensorFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
DECLARE_DISPATCHER(BitwiseAndTensorFn, bitwise_and_tensor_dispatcher)

} // namespace at::native::flagos
