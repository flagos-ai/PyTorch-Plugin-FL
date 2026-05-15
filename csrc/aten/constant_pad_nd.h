// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "dispatch_stub.h"

namespace at::native::flagos {

using ConstantPadNdFn = at::Tensor (*)(const at::Tensor&, at::IntArrayRef, const at::Scalar&);
FLAGOS_DECLARE_DISPATCH(ConstantPadNdFn, constant_pad_nd_stub)

} // namespace at::native::flagos
