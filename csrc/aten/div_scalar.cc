// Copyright (c) 2026, BAAI. All rights reserved.

#include "div_scalar.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(DivScalarFn, div_scalar_dispatcher, "div.Scalar")

} // namespace at::native::flagos
