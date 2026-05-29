// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatcher.h"

namespace at::native::flagos {

using ConstantPadNdFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef, const at::Scalar&);
DECLARE_DISPATCHER(ConstantPadNdFn, constant_pad_nd_dispatcher)

} // namespace at::native::flagos
