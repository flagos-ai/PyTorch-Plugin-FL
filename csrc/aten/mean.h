// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using MeanDimFn = at::Tensor (*)(const at::Tensor&, at::OptionalIntArrayRef,
                                  bool, std::optional<at::ScalarType>);
DECLARE_DISPATCHER(MeanDimFn, mean_dim_dispatcher)

} // namespace at::native::flagos
