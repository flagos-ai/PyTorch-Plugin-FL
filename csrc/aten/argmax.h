// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using ArgmaxFn = at::Tensor (*)(const at::Tensor&, std::optional<int64_t>, bool);
DECLARE_DISPATCHER(ArgmaxFn, argmax_dispatcher)

using ArgminFn = at::Tensor (*)(const at::Tensor&, std::optional<int64_t>, bool);
DECLARE_DISPATCHER(ArgminFn, argmin_dispatcher)

} // namespace at::native::flagos
