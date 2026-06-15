// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../silu_backward.h"

#include <ATen/ops/silu_backward.h>

namespace at::native::flagos {

namespace {

at::Tensor SiluBackwardKernelMetax(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  const at::Tensor grad_cpu = grad_output.cpu();
  const at::Tensor self_cpu = self.cpu();
  at::Tensor out_cpu = at::silu_backward(grad_cpu, self_cpu);
  return out_cpu.to(grad_output.device(), out_cpu.scalar_type());
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    SiluBackwardFn, silu_backward_dispatcher, Backend::kMetax, SiluBackwardKernelMetax)

} // namespace at::native::flagos
