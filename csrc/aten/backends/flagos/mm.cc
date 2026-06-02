// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mm.h"

#include <flag_gems/operators.h>

namespace at::native::flagos {

namespace {

void MmKernelFlaggems(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  flag_gems::mm_out_tensor(self, mat2, out);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kFlagOs, MmKernelFlaggems)

} // namespace at::native::flagos
