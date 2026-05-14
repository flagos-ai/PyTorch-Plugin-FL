// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using SumDimFn = at::Tensor (*)(const at::Tensor&, at::OptionalIntArrayRef,
                                bool, std::optional<at::ScalarType>);
FLAGOS_DECLARE_DISPATCH(SumDimFn, sum_dim_stub)

} // namespace at::native::flagos
