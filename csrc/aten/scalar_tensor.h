// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Scalar.h>
#include "dispatcher.h"

namespace at::native::flagos {

using ScalarTensorFn = at::Tensor (*)(const at::Scalar&,
                                      std::optional<at::ScalarType>,
                                      std::optional<at::Layout>,
                                      std::optional<at::Device>,
                                      std::optional<bool>);
DECLARE_DISPATCHER(ScalarTensorFn, scalar_tensor_dispatcher)

} // namespace at::native::flagos
