// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using CatFn = at::Tensor (*)(const at::ITensorListRef&, int64_t);
FLAGOS_DECLARE_DISPATCH(CatFn, cat_stub)

} // namespace at::native::flagos
