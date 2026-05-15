// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using PowTensorScalarFn = at::Tensor (*)(const at::Tensor&, const at::Scalar&);
FLAGOS_DECLARE_DISPATCH(PowTensorScalarFn, pow_tensor_scalar_stub)

} // namespace at::native::flagos
