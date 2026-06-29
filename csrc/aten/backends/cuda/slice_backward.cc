// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../slice_backward.h"

#include <ATen/ops/slice_scatter.h>
#include <ATen/ops/zeros.h>

#include "../../device_boxing.h"

namespace at::native::flagos {

namespace {

at::Tensor SliceBackwardKernelCuda(
    const at::Tensor& grad_output, at::IntArrayRef input_sizes,
    int64_t dim, int64_t start, int64_t end, int64_t step) {
  // Create grad_input as CUDA zeros, box grad_output to CUDA,
  // then call slice_scatter on CUDA tensors.
  auto cuda_options = grad_output.options().device(c10::DeviceType::CUDA);
  auto grad_input = at::zeros(input_sizes, cuda_options);

  BoxToCuda(grad_output);
  auto result = at::slice_scatter(grad_input, grad_output, dim, start, end, step);
  UnboxToFlagos(grad_output);
  UnboxToFlagos(result);
  return result;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SliceBackwardFn, slice_backward_dispatcher, Backend::kCuda, SliceBackwardKernelCuda)

} // namespace at::native::flagos
