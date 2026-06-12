// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Generator.h>
#include "dispatcher.h"

namespace at::native::flagos {

using MultinomialFn = at::Tensor (*)(const at::Tensor&, int64_t, bool, ::std::optional<at::Generator>);
DECLARE_DISPATCHER(MultinomialFn, multinomial_dispatcher)

} // namespace at::native::flagos
