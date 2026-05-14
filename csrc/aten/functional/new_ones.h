// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using NewOnesFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef,
                                 std::optional<at::ScalarType>,
                                 std::optional<at::Layout>,
                                 std::optional<at::Device>,
                                 std::optional<bool>);
FLAGOS_DECLARE_DISPATCH(NewOnesFn, new_ones_stub)

} // namespace at::native::flagos
