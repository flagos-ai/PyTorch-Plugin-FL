// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using ZerosFn = at::Tensor (*)(at::IntArrayRef,
                               std::optional<at::ScalarType>,
                               std::optional<at::Layout>,
                               std::optional<at::Device>,
                               std::optional<bool>);
FLAGOS_DECLARE_DISPATCH(ZerosFn, zeros_stub)

} // namespace at::native::flagos
