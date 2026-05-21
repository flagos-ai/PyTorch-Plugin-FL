// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../slice_backward.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

at::Tensor SliceBackwardKernelAscend(const at::Tensor& grad_output, at::IntArrayRef input_sizes,
                                     int64_t dim, int64_t start, int64_t end, int64_t step) {
  namespace ascend = at::native::flagos::ascend;
  auto grad_input = ascend::OpPreparation::apply_tensor_without_format(
      input_sizes, grad_output.options());
  grad_input.zero_();

  auto slice = grad_input.slice(dim, start, end, step);
  slice.copy_(grad_output);

  return grad_input;
}

FLAGOS_REGISTER_DISPATCH(SliceBackwardFn, slice_backward_stub, FlagosDevice::kAscend, SliceBackwardKernelAscend)

} // namespace at::native::flagos
