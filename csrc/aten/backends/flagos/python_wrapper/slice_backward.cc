// Copyright (c) 2026, BAAI. All rights reserved.

#include "../python_op_caller.h"
#include "../../../slice_backward.h"

#include <ATen/core/ivalue.h>

namespace at::native::flagos {

namespace {

at::Tensor SliceBackwardKernelPython(const at::Tensor& grad_output, at::IntArrayRef input_sizes,
                                     int64_t dim, int64_t start, int64_t end, int64_t step) {
  std::vector<c10::IValue> args;
  args.emplace_back(grad_output);
  args.emplace_back(input_sizes.vec());
  args.emplace_back(dim);
  args.emplace_back(start);
  args.emplace_back(end);
  args.emplace_back(step);
  return CallPythonOp_Generic("slice_backward", args);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(SliceBackwardFn, slice_backward_dispatcher, Backend::kFlagOsPython, SliceBackwardKernelPython)

} // namespace at::native::flagos
