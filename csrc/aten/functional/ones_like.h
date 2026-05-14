// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using OnesLikeFn = at::Tensor (*)(const at::Tensor&,
                                  std::optional<at::ScalarType>,
                                  std::optional<at::Layout>,
                                  std::optional<at::Device>,
                                  std::optional<bool>,
                                  std::optional<at::MemoryFormat>);
FLAGOS_DECLARE_DISPATCH(OnesLikeFn, ones_like_stub)

} // namespace at::native::flagos
