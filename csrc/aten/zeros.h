// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using ZerosFn = at::Tensor (*)(at::IntArrayRef,
                               std::optional<at::ScalarType>,
                               std::optional<at::Layout>,
                               std::optional<at::Device>,
                               std::optional<bool>);
DECLARE_DISPATCHER(ZerosFn, zeros_dispatcher)

} // namespace at::native::flagos
