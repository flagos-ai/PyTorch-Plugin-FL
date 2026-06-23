// Copyright (c) 2026, BAAI. All rights reserved.

#include "topk.h"

namespace at::native::flagos {

ADD_IMPL_TO_DISPATCHER(TopkFn, topk_dispatcher, "topk")

} // namespace at::native::flagos
