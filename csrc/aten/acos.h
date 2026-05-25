// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using AcosFn = at::Tensor (*)(const at::Tensor&);
FLAGOS_DECLARE_DISPATCH(AcosFn, acos_stub)

} // namespace at::native::flagos
