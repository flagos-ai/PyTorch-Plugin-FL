// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "dispatcher.h"

namespace at::native::flagos {

using MulScalarFn = at::Tensor (*)(const at::Tensor&, const at::Scalar&);
DECLARE_DISPATCHER(MulScalarFn, mul_scalar_dispatcher)

} // namespace at::native::flagos
