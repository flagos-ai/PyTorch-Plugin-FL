// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using MulScalarFn = at::Tensor (*)(const at::Tensor&, const at::Scalar&);
FLAGOS_DECLARE_DISPATCH(MulScalarFn, mul_scalar_stub)

} // namespace at::native::flagos
