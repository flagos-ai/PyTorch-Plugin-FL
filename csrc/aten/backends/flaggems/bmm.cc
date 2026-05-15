// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

void BmmKernelFlaggems(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  auto result = flag_gems::bmm(self, mat2);
  out.copy_(result);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(BmmFn, bmm_stub, FlagosDevice::kFlagOs, BmmKernelFlaggems)

} // namespace at::native::flagos
