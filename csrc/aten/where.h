// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using WhereSelfFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&, const at::Tensor&);
DECLARE_DISPATCHER(WhereSelfFn, where_self_dispatcher)

} // namespace at::native::flagos
