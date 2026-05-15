// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using AddTensorFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Scalar&);
FLAGOS_DECLARE_DISPATCH(AddTensorFn, add_tensor_stub)

} // namespace at::native::flagos
