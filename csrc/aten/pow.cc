// Copyright (c) 2026, BAAI. All rights reserved.

#include "pow.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(PowTensorScalarFn, pow_tensor_scalar_dispatcher, "pow.Tensor_Scalar")

} // namespace at::native::flagos
