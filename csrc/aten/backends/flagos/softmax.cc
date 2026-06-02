// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../softmax.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

at::Tensor SoftmaxKernelFlaggems(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return flag_gems::softmax(self, dim, half_to_float);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SoftmaxFn, softmax_dispatcher, Backend::kFlagOs, SoftmaxKernelFlaggems)

} // namespace at::native::flagos
