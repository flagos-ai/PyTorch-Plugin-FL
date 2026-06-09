// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../constant_pad_nd.h"

#include <ATen/ops/constant_pad_nd.h>

namespace at::native::flagos {

namespace {

at::Tensor ConstantPadNdKernelMetax(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value) {
  const at::Tensor self_cpu = self.cpu();
  at::Tensor out_cpu = at::constant_pad_nd(self_cpu, pad, value);
  return out_cpu.to(self.device(), out_cpu.scalar_type());
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    ConstantPadNdFn, constant_pad_nd_dispatcher, Backend::kMetax, ConstantPadNdKernelMetax)

} // namespace at::native::flagos
