// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using SiluFn = at::Tensor (*)(const at::Tensor&);
DECLARE_DISPATCHER(SiluFn, silu_dispatcher)

} // namespace at::native::flagos
