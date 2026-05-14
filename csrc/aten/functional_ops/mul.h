// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using MulTensorFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
FLAGOS_DECLARE_DISPATCH(MulTensorFn, mul_tensor_stub)

} // namespace at::native::flagos
