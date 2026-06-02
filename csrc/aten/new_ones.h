// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using NewOnesFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef,
                                 std::optional<at::ScalarType>,
                                 std::optional<at::Layout>,
                                 std::optional<at::Device>,
                                 std::optional<bool>);
DECLARE_DISPATCHER(NewOnesFn, new_ones_dispatcher)

} // namespace at::native::flagos
