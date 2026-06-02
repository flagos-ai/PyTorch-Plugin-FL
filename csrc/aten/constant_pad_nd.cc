// Copyright (c) 2026, BAAI. All rights reserved.

#include "constant_pad_nd.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(ConstantPadNdFn, constant_pad_nd_dispatcher, "constant_pad_nd")

} // namespace at::native::flagos
