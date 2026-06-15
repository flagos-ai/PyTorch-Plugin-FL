// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../slice_backward.h"

#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/zeros.h>

namespace at::native::flagos {

namespace {

at::Tensor SliceBackwardKernelMetax(
    const at::Tensor& grad_output,
    at::IntArrayRef input_sizes,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  const at::Tensor grad_cpu = grad_output.cpu();
  at::Tensor grad_input = at::zeros(input_sizes, grad_cpu.options());
  at::Tensor out_cpu =
      at::slice_scatter(grad_input, grad_cpu, dim, start, end, step);
  return out_cpu.to(grad_output.device(), out_cpu.scalar_type());
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    SliceBackwardFn, slice_backward_dispatcher, Backend::kMetax, SliceBackwardKernelMetax)

} // namespace at::native::flagos
