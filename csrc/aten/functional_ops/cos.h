// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using CosFn = at::Tensor (*)(const at::Tensor&);
FLAGOS_DECLARE_DISPATCH(CosFn, cos_stub)

} // namespace at::native::flagos
