// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../slice_backward.h"

#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/zeros.h>

namespace at::native::flagos {

namespace {

at::Tensor SliceBackwardKernelCuda(
    const at::Tensor& grad_output, at::IntArrayRef input_sizes,
    int64_t dim, int64_t start, int64_t end, int64_t step) {
  auto grad_input = at::zeros(input_sizes, grad_output.options());
  return at::slice_scatter(grad_input, grad_output, dim, start, end, step);
}

} // namespace

FLAGOS_REGISTER_DISPATCH(SliceBackwardFn, slice_backward_stub, FlagosDevice::kCuda, SliceBackwardKernelCuda)

} // namespace at::native::flagos
