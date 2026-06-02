// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using CatFn = at::Tensor (*)(const at::ITensorListRef&, int64_t);
DECLARE_DISPATCHER(CatFn, cat_dispatcher)

} // namespace at::native::flagos
